# TODO: Add ability for synapses to affect rxd concentration

# TODO: How to handle conductance for DA channel, we have that DA vesicles contain 33000 DA molecules
#       (watch out for overflow of integers in snudda synapse matrix)

# TODO: External input should also be able to release dopamine

import os
import numpy as np
import neuron.crxd as rxd
import json
from itertools import chain


class NeuronModulation:

    def __init__(self, neuron, config_file=None):

        self.neuron = neuron
        self.compartments = dict()
        self.species = dict()
        self.rates = dict()
        self.reactions = dict()
        self.config_data = {}
        self.config_file = config_file

        self.node_cache = None

        self.build = {"soma_internal": lambda neuron_dummy: self.set_default_compartments("soma", nrn_region="i"),
                      "dend_internal": lambda neuron_dummy: self.set_default_compartments("dend", nrn_region="i"),
                      "axon_internal": lambda neuron_dummy: self.set_default_compartments("axon", nrn_region="i"),
                      "soma_external": lambda neuron_dummy: self.set_default_compartments("soma", nrn_region="o"),
                      "dend_external": lambda neuron_dummy: self.set_default_compartments("dend", nrn_region="o"),
                      "axon_external": lambda neuron_dummy: self.set_default_compartments("axon", nrn_region="o") }

    def __del__(self):

        # Clear old rxd objects -- this avoids segfault for unittests
        rxd.rxd.byeworld()

    def set_default_compartments(self, compartment, nrn_region="i"):
        if compartment == "axon":
            section_list = self.neuron.icell.axon
        elif compartment == "dend":
            section_list = self.neuron.icell.dend
        elif compartment == "soma":
            section_list = self.neuron.icell.soma

        return rxd.Region(section_list, nrn_region=nrn_region)

    def add_concentration(self, species_name, diffusion_constant, initial_conc,
                          charge=0,
                          compartment=("soma_internal", "dend_internal")):

        if species_name not in self.species:
            self.species[species_name] = dict()

        for comp in compartment:
            if comp not in self.compartments:
                self.compartments[comp] = self.build[comp](self.neuron)

            if comp in self.species[species_name]:
                raise ValueError(f"{species_name = } already defined for {comp = }")

            # TODO: Add atol_scale etc...
            self.species[species_name][comp] = rxd.Species(self.compartments[comp],
                                                           d=diffusion_constant,
                                                           initial=initial_conc,
                                                           charge=charge,
                                                           name=species_name)

    def get_species(self, *species, region_name):
        """ Example usage:
         a, b, c = self.neurons[123].modulation.get_species('A', 'B', 'C')
         """
        return [self.species[x][region_name] for x in species]

    def add_decay(self, species_name, decay_rate):

        species = self.species[species_name]
        self.rates[f"decay_{species_name}"] = rxd.rate(species, -decay_rate*species)

    def add_rate(self, species_name, left_side, right_side, region_name, overwrite=False):

        if species_name not in self.rates:
            self.rates[species_name] = dict()

        if not overwrite and region_name in self.rates[species_name]:
            raise KeyError(f"Reaction {species_name} is already defined in neuron {self.neuron.name}")

        self.rates[species_name][region_name] = rxd.Rate(left_side, right_side,
                                                         regions=self.compartments[region_name])

    def get_neuron_regions(self, region_list):

        regions = []
        for region in region_list:
            regions.append(self.compartments[region])

        return chain(*regions)

    def add_reaction(self, reaction_name, left_side, right_side, forward_rate, backward_rate, region_name, overwrite=False):

        if reaction_name not in self.reactions:
            self.reactions[reaction_name] = dict()

        if not overwrite and region_name in self.reactions[reaction_name]:
            raise KeyError(f"Reaction {reaction_name} is already defined in neuron {self.neuron.name}")

        if backward_rate is None:
            backward_rate = 0

        self.reactions[reaction_name][region_name] = rxd.Reaction(left_side,
                                                                  right_side,
                                                                  forward_rate,
                                                                  backward_rate,
                                                                  regions=self.compartments[region_name])

    def build_node_cache(self):

        print(f"Build node cache")
        self.node_cache = {}

        for species_name, species_data in self.species.items():

            if species_name not in self.node_cache:
                self.node_cache[species_name] = {}

            for region_name, species in species_data.items():

                if region_name not in self.node_cache[species_name]:
                    self.node_cache[species_name][region_name] = {}

                # This row below might need to be sped up
                all_nodes = species.nodes

                for node in all_nodes:
                    if node._sec._sec not in self.node_cache[species_name][region_name]:
                         self.node_cache[species_name][region_name][node._sec._sec] = ([], [])

                    self.node_cache[species_name][region_name][node._sec._sec][0].append(node)
                    self.node_cache[species_name][region_name][node._sec._sec][1].append(node._location)

                # Now we want to also extract the sec_x
                for node_key in self.node_cache[species_name][region_name]:
                    node_list, node_x = self.node_cache[species_name][region_name][node_key]
                    self.node_cache[species_name][region_name][node_key] = (node_list, np.array(node_x))

        print(f"Node cache built.")

    def get_node_from_cache(self, species_name, seg, region_name):
        node_list, node_x = self.node_cache[species_name][region_name][seg.sec]
        idx = np.argmin(np.abs(node_x - seg.x))
        return node_list[idx]

    def clear_cache(self):
        self.node_cache = None

    def link_synapse(self, species_name, region: str, synapse, flux_variable: str):
        # region: "soma_internal", "soma_external", "dend_internal", "dend_external"

        # print(f"link_synapses {species_name}, {region}")

        if self.node_cache is None:
            self.build_node_cache()

        node = self.get_node_from_cache(species_name=species_name, seg=synapse.get_segment(), region_name=region)
        node.include_flux(synapse, flux_variable)

        # self.species[species_name][region].nodes(synapse.get_segment())[0].include_flux(synapse, flux_variable)

    def load_json(self, config_path=None):

        print(f"Parsing neuromodulation json: {config_path}")

        if config_path is None:
            config_path = self.config_file

        with open(config_path, "r") as f:
            self.config_data = json.load(f)

        print(f"Parsing species")

        for species_name, species_data in self.config_data.get("species", {}).items():
            initial_concentration = species_data.get("initial_concentration", 0)
            diffusion_constant = species_data.get("diffusion_constant", 0)
            charge = species_data.get("charge", 0)
            regions = species_data.get("regions", ("soma_internal", "dendrites_internal"))
            # TODO: Read atol_scale, ecs_boundary_boundary_conditions, represents parameters

            self.add_concentration(species_name=species_name,
                                   diffusion_constant=diffusion_constant,
                                   initial_conc=initial_concentration,
                                   compartment=regions, charge=charge)

        # Black magic, setup the species variables
        species_name_vars = ",".join(self.species.keys())
        species_name_str = "','".join(self.species.keys())

        print(f"Parsing rates.")

        for rate_name, rate_data in self.config_data.get("rates", {}).items():
            if rate_name not in self.species:
                raise ValueError(f"Species {rate_name} is not defined. Available: {self.species.keys()}")

            rates = rate_data["rates"]
            if isinstance(rates, str) and len(rate_data["regions"]) > 1:
                rates = [rates for x in rate_data["regions"]]

            for region, rate in zip(rate_data["regions"], rates):
                exec(f"{species_name_vars} = self.get_species('{species_name_str}', region_name=region)")
                right_side = eval(rate)

                self.add_rate(species_name=rate_name,
                              left_side=self.get_species(rate_name, region_name=region)[0],
                              right_side=right_side,
                              region_name=region)

        print(f"Parsing reactions")

        for reaction_name, reaction_data in self.config_data.get("reactions", {}).items():

            forward_rates = reaction_data["forward_rate"]
            backward_rates = reaction_data["backward_rate"]

            if not isinstance(forward_rates, (tuple, list)):
                forward_rates = [forward_rates] * len(reaction_data["regions"])

            if not isinstance(backward_rates, (tuple, list)):
                backward_rates = [backward_rates] * len(reaction_data["regions"])

            if not ( len(reaction_data["regions"]) == len(forward_rates) == len(backward_rates)):
                raise ValueError(f"{reaction_data} incompatible lengths for regions, forward and backward rates")

            for region, forward_rate, backward_rate in zip(reaction_data["regions"], forward_rates, backward_rates):
                exec(f"{species_name_vars} = self.get_species('{species_name_str}', region_name=region)")

                left_side = eval(reaction_data["reactants"])
                right_side = eval(reaction_data["products"])

                self.add_reaction(reaction_name=reaction_name,
                                  left_side=left_side,
                                  right_side=right_side,
                                  forward_rate=forward_rate,
                                  backward_rate=backward_rate,
                                  region_name=region)

        print(f"Parsing done.")
