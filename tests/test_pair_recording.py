import os
import unittest
import json
import numpy as np
from snudda.simulate.pair_recording import PairRecording
from snudda.utils.load import SnuddaLoad
from snudda.utils.load_network_simulation import SnuddaLoadNetworkSimulation


class PairRecordingTestCase(unittest.TestCase):

    def setUp(self):

        # Temporarily disable the creation of network while testing...
        return

        self.network_path = os.path.join("networks", "pair_recording_test")
        rc = None

        if os.path.isdir("x86_64"):
            import shutil
            shutil.rmtree("x86_64")
        os.system(f"nrnivmodl {os.path.join('validation', 'mechanisms')}")

        from snudda import SnuddaPlace
        from snudda import SnuddaDetect
        from snudda import SnuddaPrune

        sp = SnuddaPlace(network_path=self.network_path, rc=rc)
        sp.place()

        sd = SnuddaDetect(network_path=self.network_path, rc=rc)
        sd.detect()

        sp = SnuddaPrune(network_path=self.network_path, rc=rc)
        sp.prune()

        self.experiment_config_file = os.path.join(self.network_path, "experiment.json")
        network_file = os.path.join(self.network_path, "network-synapses.hdf5")
        self.pr = PairRecording(network_path=self.network_path, network_file=network_file,
                                experiment_config_file=self.experiment_config_file)
        self.pr.run()

    def test_frequency(self):

        return

        # TODO: TEMP line, this is normally done in setUp... remove later
        self.network_path = os.path.join("networks", "pair_recording_test")
        self.experiment_config_file = os.path.join(self.network_path, "experiment.json")

        sim_file = os.path.join(self.network_path, "simulation", "pair-recording-simulation.hdf5")

        sl = SnuddaLoad(self.network_path)
        sns = SnuddaLoadNetworkSimulation(network_path=self.network_path,
                                          network_simulation_output_file=sim_file)

        with open(self.experiment_config_file, "r") as f:
            experiment_config = json.load(f)

        for inj_info in experiment_config["currentInjection"]:

            neuron_id = inj_info["neuronID"]
            start = inj_info["start"]
            end = inj_info["end"]
            requested_freq = inj_info["requestedFrequency"]

            with self.subTest(f"Testing neuron {neuron_id}"):

                spikes = sns.get_spikes(neuron_id)
                for s, e in zip(start, end):
                    n_spikes = len(np.where(np.logical_and(s <= spikes, spikes <= e))[0])
                    freq = n_spikes / (e - s)

                    print(f"neuron_id: {neuron_id}, requested_freq: {requested_freq} Hz, actual freq: {freq}")
                    self.assertTrue(requested_freq * 0.8 < freq < requested_freq * 1.2,
                                    f"neuron_id: {neuron_id}, requested_freq: {requested_freq} Hz, actual freq: {freq}")

#        import pdb
#        pdb.set_trace()

        # TODO: Read in the frequency of each neuron. Compare the simulated frequency to the requested frequency

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()