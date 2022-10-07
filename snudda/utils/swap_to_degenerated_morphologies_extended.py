import numpy as np

import json
from collections import OrderedDict

from snudda import SnuddaLoad
from snudda.utils.swap_to_degenerated_morphologies import SwapToDegeneratedMorphologies

# TODO: We need to reapply the pruning rule to all neurons...


class SwapToDegeneratedMorphologiesExtended(SwapToDegeneratedMorphologies):

    def __init__(self, original_network_file, updated_network_file, output_network_file,
                 original_snudda_data_dir, updated_snudda_data_dir,
                 original_input_file=None, updated_input_file=None, output_input_file=None,
                 filter_axon=True,
                 random_seed=None
                 ):

        """
            Args:
                original_network_file : Original network-synapses.hdf5 (e.g. network with WT morphologies)
                updated_network_file : network-synapses.hdf5 network with updated morphologies (e.g. Parkinson network)
                output_network_file : Created network-synapses.hdf5 network by this code
                original_snudda_data_dir : Path to original data (e.g. WT SNUDDA_DATA)
                updated_snudda_data_dir : Path to updated data (e.g. Parkinson SNUDDA_DATA)
                original_input_file : Input to original network (e.g. WT network input)
                updated_input_file : Input to the updated_network_file network
                output_input_file: Input file generated by this code (great name!)
                filter_axon : Degeneration of axon leads to removal of synapses
                random_seed: Random seed used for post pruning

        """

        super().__init__(original_network_file=original_network_file,  # PD0
                         new_network_file=output_network_file,         # degenerated PD2 without FS axon
                         original_snudda_data_dir=original_snudda_data_dir,
                         new_snudda_data_dir=updated_snudda_data_dir,
                         original_input_file=original_input_file,
                         new_input_file=output_input_file, filter_axon=filter_axon)

        self.updated_network_file = updated_network_file  # PD2 fresh touch detection, generated before
        self.updated_network_loader = SnuddaLoad(self.updated_network_file, load_synapses=False)
        self.updated_hdf5 = self.updated_network_loader.hdf5_file
        self.updated_data = self.updated_network_loader.data
        self.random_seed = random_seed

        self.old_kd_tree_cache = dict()
        self.old_neuron_cache_id = dict()

        self.check_same_network()
        self.rng = np.random.default_rng(seed=random_seed)

    def close(self):

        super().close()

        if self.updated_hdf5:
            self.updated_hdf5.close()

    def check_same_network(self):

        assert len(self.original_network_loader.data["neurons"]) == len(self.updated_network_loader.data["neurons"]),\
            f"Original and updated network does not have the same number of neurons!"

        for orig_neuron, updated_neuron in zip(self.original_network_loader.data["neurons"],
                                               self.updated_network_loader.data["neurons"]):

            assert orig_neuron["neuronID"] == updated_neuron["neuronID"], f"Internal error, neuron ID mismatch"

            assert orig_neuron["name"] == updated_neuron["name"], \
                (f"Name mismatch for neuron {orig_neuron['neuronID']}: {orig_neuron['name']} {updated_neuron['name']}"
                 f"\nDid you use the same random seed when calling init to generate the networks?")

            assert (orig_neuron["position"] == updated_neuron["position"]).all(), \
                (f"Position mismatch for neuron {orig_neuron['neuronID']}: "
                 f"{orig_neuron['position']} {updated_neuron['position']}"
                 f"\nDid you use the same random seed when calling init to generate the networks?")

            assert (orig_neuron["rotation"] == updated_neuron["rotation"]).all(), \
                (f"Position mismatch for neuron {orig_neuron['neuronID']}: " 
                 f"{orig_neuron['rotation']} {updated_neuron['rotation']}"
                 f"\nDid you use the same random seed when calling init to generate the networks?")

    def get_degeneration_recovery_lookups(self, network_config):

        type_lookup = self.updated_network_loader.get_neuron_types()
        degeneration_recovery = dict()

        for conf_key, conf_info in network_config["Connectivity"].items():
            pre_type, post_type = conf_key.split(",")

            for con_type, con_info in conf_info.items():
                channel_model_id = con_info["channelModelID"]

                if "degenerationRecovery" in con_info["pruning"]:
                    degeneration_recovery[pre_type, post_type, channel_model_id] = con_info["pruning"]["degenerationRecovery"]

        return degeneration_recovery, type_lookup

    def get_additional_synapses(self, network_config, rng, synapse_distance_treshold=None):

        # Calculate coordinate remapping for updated synapses

        voxel_size = self.updated_network_loader.data["voxelSize"]
        assert voxel_size == self.original_network_loader.data["voxelSize"], f"Voxel size mismatch between networks"

        if synapse_distance_treshold is None:
            synapse_distance_treshold = 1.2*(3 * voxel_size ** 2) ** 0.5  # 5.2e-6, maximal mismatch due to moving origos, actually 5.4 micrometers

        orig_sim_origo = self.original_network_loader.data["simulationOrigo"]
        updated_sim_origo = self.updated_network_loader.data["simulationOrigo"]

        origo_diff = updated_sim_origo - orig_sim_origo
        voxel_transform = np.round(origo_diff / voxel_size)

        # Will hold indexes into synapse matrix, so we can do all synapses belonging to
        # one morphology all at once
        pre_neuron_synapses = dict()
        post_neuron_synapses = dict()

        for nid in self.updated_network_loader.get_neuron_id():
            pre_neuron_synapses[nid] = []
            post_neuron_synapses[nid] = []

        # All the synapses from the fresh PD2 touch detection
        synapse_matrix = self.updated_network_loader.data["synapses"][()].copy()

        keep_mask = np.zeros((synapse_matrix.shape[0],), dtype=bool)

        for idx, synapse_row in enumerate(synapse_matrix):
            pre_neuron_synapses[synapse_row[0]].append(idx)
            post_neuron_synapses[synapse_row[1]].append(idx)

        for nid in self.updated_network_loader.get_neuron_id():
            if nid % 100 == 0:
                print(f"Processing neuron {nid}")

            pre_idx = np.array(pre_neuron_synapses[nid], dtype=int)
            post_idx = np.array(post_neuron_synapses[nid], dtype=int)

            pre_coords = synapse_matrix[pre_idx, 2:5] * voxel_size + updated_sim_origo
            post_coords = synapse_matrix[post_idx, 2:5] * voxel_size + updated_sim_origo

            morph = self.get_morphology(neuron_id=nid, hdf5=self.old_hdf5,
                                        neuron_cache_id=self.old_neuron_cache_id,
                                        snudda_data=self.original_snudda_data_dir)

            dend_kd_tree = self.get_kd_tree(morph, "dend", kd_tree_cache=self.old_kd_tree_cache)
            synapse_dend_dist, _ = dend_kd_tree.query(post_coords)
            syn_mask = np.logical_and(synapse_dend_dist > synapse_distance_treshold, np.linalg.norm(post_coords - morph.soma[0, :3], axis=1) > morph.soma[0, 3] + synapse_distance_treshold)
            keep_mask[post_idx[np.where(syn_mask)[0]]] = True

            if self.updated_network_loader.data["neurons"][nid]["axonDensity"] is None:
                try:
                    axon_kd_tree = self.get_kd_tree(morph, "axon", kd_tree_cache=self.old_kd_tree_cache)
                    synapse_axon_dist, _ = axon_kd_tree.query(pre_coords)
                    keep_mask[pre_idx[np.where(synapse_axon_dist > synapse_distance_treshold)[0]]] = True

                    # if nid == 100:
                    #     print("Tell me why .. axon")
                    #     import pdb
                    #     pdb.set_trace()

                except:
                    import traceback
                    print(traceback.format_exc())
                    import pdb
                    pdb.set_trace()
            else:
                print(f"No axon for neuron {morph.name} ({nid})")

        print("Synapse degeneration recovery...")
        # If degenerationRecovery is set in the pruning rules, we will add that fraction of the synapses on the
        # overlapping morphology part back to the network synapses -- and take care not to place synapses
        # exactly where old synapses are already placed

        # Get information about synapse degeneration recovery (ie how large a fraction of the synapses
        # that are in the overlaping part of the dendrites should be kept to recover a little...
        degeneration_recovery, type_lookup = self.get_degeneration_recovery_lookups(network_config=network_config)

        if len(degeneration_recovery) > 0:
            p_recovery = np.zeros((synapse_matrix.shape[0],), dtype=float)

            for idx, syn_row in enumerate(synapse_matrix):
                pre_type = type_lookup[syn_row[0]]
                post_type = type_lookup[syn_row[1]]
                channel_model_id = syn_row[6]

                if (pre_type, post_type, channel_model_id) in degeneration_recovery:
                    p_recovery[idx] = degeneration_recovery[pre_type, post_type, channel_model_id]

            recovery_mask = p_recovery > rng.random(p_recovery.shape)
            keep_mask = np.logical_or(keep_mask, recovery_mask)

        # Transform coordinates to new simulation origo
        added_synapses = synapse_matrix[keep_mask, :].copy()
        added_synapses[:, 2:5] = added_synapses[:, 2:5] + voxel_transform

        return added_synapses

    def sort_synapses(self, synapses):

        # Sort order: columns 1 (dest), 0 (src), 6 (synapse type)
        sort_idx = np.lexsort(synapses[:, [6, 0, 1]].transpose())
        return synapses[sort_idx, :].copy()

    def filter_synapses(self, filter_axon=False, post_degen_pruning=True):

        # This replaces the original filter_synapses, so that we can also add in the
        # new synapses due to growing axons or dendrites

        config = json.loads(self.updated_network_loader.data["config"], object_pairs_hook=OrderedDict)

        # This needs to be made bigger!
        num_rows = self.old_hdf5["network/synapses"].shape[0] + self.updated_hdf5["network/synapses"].shape[0]
        num_cols = self.old_hdf5["network/synapses"].shape[1]
        new_synapses = np.zeros((num_rows, num_cols), dtype=self.old_hdf5["network/synapses"].dtype)
        syn_ctr = 0

        # Keep synapses in original network that are still within the new morphologies
        for synapses in self.synapse_iterator():

            # if synapses[0, 0] == 4 and synapses[0, 1] == 100:
            #     print("tell me why...")
            #     import pdb
            #     pdb.set_trace()

            new_syn = self.filter_synapses_helper(synapses, filter_axon=filter_axon)                    
            new_synapses[syn_ctr:syn_ctr + new_syn.shape[0]] = new_syn
            syn_ctr += new_syn.shape[0]

        # Here add the new synapses from growing axons and dendrites
        additional_synapses = self.get_additional_synapses(network_config=config, rng=self.rng)
        new_synapses[syn_ctr:syn_ctr+additional_synapses.shape[0], :] = additional_synapses
        syn_ctr += additional_synapses.shape[0]

        # Resort the new synapse matrix
        sorted_synapses = self.sort_synapses(new_synapses[:syn_ctr, :])

        old_synapse_iterator = self.synapse_iterator(synapses=self.old_hdf5["network/synapses"][()])

        if post_degen_pruning:
            pruned_synapses = self.post_degeneration_pruning(synapses=sorted_synapses,
                                                             old_synapse_iterator=old_synapse_iterator,
                                                             network_config=config,
                                                             rng=self.rng)
        else:
            pruned_synapses = sorted_synapses

        sorted_synapses = None

        # TODO: We should check that the addition of "degeneratio recovery synapses" are not duplicates
        #       of the PD degeneration synapses (this might happen in 8% of the cases...)

        self.new_hdf5["network"].create_dataset("synapses", data=pruned_synapses, compression="lzf")

        num_synapses = np.zeros((1,), dtype=np.uint64)
        self.new_hdf5["network"].create_dataset("nSynapses", data=pruned_synapses.shape[0], dtype=np.uint64)

        print(f"Keeping {self.new_hdf5['network/nSynapses'][()]} "
              f"out of {self.old_hdf5['network/nSynapses'][()]} synapses "
              f"({self.new_hdf5['network/nSynapses'][()] / self.old_hdf5['network/nSynapses'][()]*100:.3f} %)")

    def post_degeneration_pruning(self, synapses, old_synapse_iterator, network_config, rng):

        """ Args:
            synapses: Synapse matrix to prune
            old_synapse_iterator: Iterator over the original synapses
            network_config: The config as a dictionary
            rng: Numpy random generator
        """

        # TODO: We here assume that mu2 is the same for WT and degenerate network, if it is NOT the same
        #       this code needs to be modified such that mu2_original and mu2_degenerate are used

        # This prunes the network synapses after the generation/growth phase
        print("Running post degeneration pruning of synapses", flush=True)

        keep_synapse_flag = np.ones((synapses.shape[0]), dtype=bool)

        type_lookup = self.updated_network_loader.get_neuron_types()
        mu2_lookup = dict()
        n_neurons = self.updated_network_loader.data["nNeurons"]

        for conf_key, conf_info in network_config["Connectivity"].items():
            pre_type, post_type = conf_key.split(",")

            if pre_type not in mu2_lookup:
                mu2_lookup[pre_type] = dict()

            if post_type not in mu2_lookup[pre_type]:
                mu2_lookup[pre_type][post_type] = dict()

            for con_type, con_info in conf_info.items():

                channel_model_id = con_info["channelModelID"]

                # OBS, this does not take into account population units. Neurons belonging to different population unit
                # can have a different mu2, but we have never used that option.
                mu2_lookup[pre_type][post_type][channel_model_id] = con_info["pruning"]["mu2"]

        synapse_ctr = 0

        old_synapse_set = next(old_synapse_iterator, None)

        p_mu_overflow = []

        for synapse_set in self.synapse_iterator(synapses=synapses):

            pre_id = synapse_set[0, 0]
            post_id = synapse_set[0, 1]
            channel_model_id = synapse_set[0, 6]

            assert (synapse_set[:, 0] == pre_id).all()
            assert (synapse_set[:, 1] == post_id).all()
            assert (synapse_set[:, 6] == channel_model_id).all(), \
                f"Code is written with assumption that all synapses between a pair of neurons are of the same type"

            old_pre_id = old_synapse_set[0, 0]
            old_post_id = old_synapse_set[0, 1]
            old_channel_mod_id = old_synapse_set[0, 6]

            while old_synapse_set is not None and old_post_id * n_neurons + old_pre_id < post_id * n_neurons + pre_id:
                old_synapse_set = next(old_synapse_iterator, None)

                if old_synapse_set is not None:
                    old_pre_id = old_synapse_set[0, 0]
                    old_post_id = old_synapse_set[0, 1]
                    old_channel_mod_id = old_synapse_set[0, 6]
                else:
                    old_pre_id = -1
                    old_post_id = -1
                    old_channel_mod_id = None

            try:
                mu2 = mu2_lookup[type_lookup[pre_id]][type_lookup[post_id]][channel_model_id]
            except:
                import traceback
                print(traceback.format_exc())
                import pdb
                pdb.set_trace()

            n_syn = synapse_set.shape[0]

            if mu2 is None:
                p_mu = 1

            else:

                if old_post_id * n_neurons + old_pre_id == post_id * n_neurons + pre_id:
                    old_n_syn = old_synapse_set.shape[0]

                    # This should be mu2 original
                    old_p_mu = 1.0 / (1.0 + np.exp(-8.0 / mu2 * (old_n_syn - mu2)))

                    assert old_channel_mod_id is None or old_channel_mod_id == channel_model_id, \
                        (f"post_degeneration_pruning: Internal error, we have assumed only one type of synapses between "
                         f"pairs of neurons in this degenerated code.\n"
                         f"channel_mod_id = {channel_model_id}, old_channel_mod_id = {old_channel_mod_id}\n"
                         f"synapse_set = {synapse_set}\n"
                         f"old_synapse_set = {old_synapse_set}")

                else:
                    old_p_mu = 1

                # This should use mu2_degenerated (which is does, but mu2_old above
                # should use mu2_original which id does not)
                # TODO: Fix it or add assert that they are the same, right now it will
                #  silently do it wrong if different...

                # We need to compensate for the old p_mu, i.e. p_real = p_mu_new / p_mu_old
                p_mu = 1.0 / (1.0 + np.exp(-8.0 / mu2 * (n_syn - mu2)))
                # print(f"p_mu = {p_mu} ({p_mu / old_p_mu}) -- {old_p_mu}")
                p_mu /= old_p_mu  # Correction factor for previous mu2 pruning

            keep_synapse_flag[synapse_ctr:synapse_ctr + n_syn] = p_mu >= rng.random()
            synapse_ctr += n_syn

            if p_mu > 1:
                p_mu_overflow.append((p_mu, n_syn))

        # Check how many synapses we are expected to be short...
        n_short = 0
        for pm, ns in p_mu_overflow:
            n_short += (pm - 1) * ns

        print(f"Unable to compensate for {n_short:.1f} degenerated synapses.")

        print(f"Post pruning. Keeping {np.sum(keep_synapse_flag)}/{len(keep_synapse_flag)} "
              f"({np.sum(keep_synapse_flag)/len(keep_synapse_flag)*100:.3f}%)")

        return synapses[keep_synapse_flag, :]


    # TODO: Profile the code to see what the bottleneck is...

    # TODO: Update filter_gap_junctions to also handle growth on dendrites

    # TODO: Also handle the new inputs that might arrive on growing morphologies...