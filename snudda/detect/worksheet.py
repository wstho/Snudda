#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 17:07:53 2025

@author: wst
"""
rng = np.random.default_rng(1)
n_targets = rng.lognormal(mean = np.log(10), sigma = 0.23, size = 100000).astype(np.int64)


sns.kdeplot(n_targets, cut = 0)


#%%
total_n_neurons = 14000
n_targets = rng.lognormal(mean = np.log(10.3), sigma = 0.23, size = 14000).astype(np.int64)

total_draws = np.sum(n_targets)
all_targets = rng.integers(0, total_n_neurons, size=total_draws)
offsets = np.cumsum(n_targets)
targets = np.split(all_targets, offsets[:-1])

#%%

sns.histplot(all_targets, binwidth = 1)
#%%


import h5py
file = '/Users/wst/Desktop/Karolinska/Simulation/Neuron/networks/skip_snudda_small_2/voxels/network-putative-synapses-91.hdf5'
with h5py.File(file) as f:
    syn = f['network/synapses'][()]
#%%

coords= sl.data['neuron_positions']

distances = squareform(pdist(coords))


n_id = 1
distances[1,: ]


#%%

from scipy.interpolate import interp1d

x = brown_ref.Distance.values
cdf = np.clip(brown_ref.Response.values, a_min = 0, a_max =1)

# Get PDF by taking differences (discrete derivative)
# pdf = np.diff(cdf) 
cdf_func = interp1d(x, cdf, kind='linear', fill_value='extrapolate')

# Get PDF by numerical derivative
dx = 10
x_fine = np.arange(x.min(), x.max(), dx)
pdf_values = np.gradient(cdf_func(x_fine), dx)


#%%

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

brown_ref = pd.read_csv('/Users/wst/Desktop/Karolinska/ReferenceData/Brown 2014/DistanceFromSoma.csv', header = None)
brown_ref.rename(columns = {0:'Distance', 1: 'Response'}, inplace = True)

x = np.concatenate([[0], brown_ref.Distance.values])
cdf = np.clip(np.concatenate([[0], brown_ref.Response.values]), a_min = 0, a_max =1)




x_mid = (x[:-1] + x[1:]) / 2  
probabilities = np.diff(cdf)



# boost_factor = np.exp(-x_mid / 500) 
# probabilities_boosted = probabilities * (1 + boost_factor * 0.5)  # Adjust 0.5 for strength
# probabilities_boosted = probabilities_boosted / probabilities_boosted.sum()  # Renormalize


center = 200
width = 200
boost_factor = 1 + 0.4 * np.exp(-((x_mid - center)**2) / (2 * width**2))  #

probabilities_boosted = probabilities * boost_factor
# probabilities_boosted = probabilities+ gaussian_bump
probabilities_boosted = probabilities_boosted / probabilities_boosted.sum()  # Renormalize


prob_lookup = interp1d(x_mid, probabilities_boosted, 
                       kind='linear', 
                       bounds_error=False, 
                       fill_value=0)



prob_lookup = interp1d(x_mid, probabilities, 
                       kind='linear', 
                       bounds_error=False, 
                       fill_value=0)



#%%
import h5py
with h5py.File('distance_lookup.hdf5', 'w') as f:
    f.create_dataset('x_mid', data=x_mid)
    f.create_dataset('probabilities', data=probabilities_boosted)
    # Save metadata
    f.attrs['kind'] = 'linear'
    f.attrs['fill_value'] = 0
# 
# # Load
# with h5py.File('distance_lookup.hdf5', 'r') as f:
#     x_mid = f['x_mid'][:]
#     probabilities = f['probabilities'][:]
#     kind = f.attrs['kind']
#     fill_value = f.attrs['fill_value']

# prob_lookup = interp1d(x_mid, probabilities, 
#                        kind=kind, bounds_error=False, fill_value=0)


#%%

# distances = squareform(pdist(sd.neuron_positions))


# n_id = 1
prob_lookup(distances[1,: ]*1e6)

dist_dep_prob = prob_lookup(distances[1,: ]*1e6)
p = dist_dep_prob/np.sum(dist_dep_prob)


#%%

max_dist = int(np.ceil(x_mid.max())) + 500  # Add buffer
lookup_array = np.zeros(max_dist)

# Fill using nearest neighbor or interpolation
distances_idx = np.arange(max_dist)
interp_func = interp1d(x_mid, probabilities, 
                       kind='linear', bounds_error=False, fill_value=0)
lookup_array = np.maximum(interp_func(distances_idx), 0)




#%%

plt.figure(figsize=(12, 4))

plt.subplot(1, 1, 1)
plt.plot(x_mid, probabilities, 'o-', label='Original')
plt.plot(x_mid, probabilities_boosted, 's-', label='Boosted')
plt.legend()
plt.show()

#%%
rng = np.random.default_rng(1)

total_n_neurons = 14000
n_conns_log = np.log(10)
all_n_conns= rng.lognormal(mean = n_conns_log, sigma = 0.25, size = total_n_neurons).astype(np.int64)

import time

t0 = time.time()
distances = squareform(pdist(coords))*1e6
print(f"Distance calc: {time.time()-t0:.2f}s")
t0 = time.time()

dist_indices = np.clip(distances.astype(int), 0, len(lookup_array)-1)
all_p = lookup_array[dist_indices]
# all_p = np.maximum(all_p, 0)
all_p= all_p / all_p.sum(axis=1, keepdims=True) ##p sum to 1 along each row

# all_p = prob_lookup(distances)
# all_p = np.maximum(all_p, 0)
# all_p= all_p / all_p.sum(axis=1, keepdims=True) ##p sum to 1 along each row

print(f"Prob lookup: {time.time()-t0:.2f}s")
t0 = time.time()

presyn = []
for n_id in range(total_n_neurons):
    presyn.append(rng.choice(total_n_neurons, size = all_n_conns[n_id], p = all_p[n_id, :], replace = False))
    
    
print(f"choice : {time.time()-t0:.2f}s")

    
#%%
##reverse engineer cdf
ds = []
for i, ps in enumerate(presyn[:]):
    ds.extend(distances[i, ps])

#%%
    
fig = plt.figure(figsize = [60*mm, 60*mm])
ax = fig.add_axes([0.2,0.2,0.75,0.75])

brown_ref = pd.read_csv('/Users/wst/Desktop/Karolinska/ReferenceData/Brown 2014/DistanceFromSoma_Thy1.csv', header = None)
brown_ref.rename(columns = {0:'Distance', 1: 'Response'}, inplace = True)
sns.lineplot(data = brown_ref, x = 'Distance', y = 'Response', ax = ax, color = 'black', lw = 3, label = 'Brown et al., 2014')

sns.ecdfplot(ds, ax = ax, alpha =1, lw = 3, label = 'simulated slice')

plt.show()
#%%
    
print(f"Distance range: {distances.min():.2f} to {distances.max():.2f}")
print(f"Prob lookup x range: {x_mid.min():.2f} to {x_mid.max():.2f}")
print(f"Negative probs: {(all_p < 0).sum()}")
print(f"Min prob value: {all_p.min()}")
    

#%%
rng = np.random.default_rng(1)

n_contacts = np.maximum(rng.lognormal(mean = np.log(10), sigma = 0.25, size = 100000).astype(np.int64), 3)
sns.histplot(n_contacts)



#%%
all_n_conns = rng.integers(10, size = 100)

presyn = []
for n_id in range(100):
    presyn.append(rng.choice(100, size = all_n_conns[n_id], replace = False))
    

#%%
presyn_id, presyn_count = np.unique(np.concatenate(presyn), return_counts = True)
represented = np.zeros(n_total_neurons, dtype=bool)

represented[presyn_id[presyn_count >= 3]] = True
unrepresented = np.where(~represented)[0]
 
   
# for u in unrepresented: 
#     for idx in rng.choice(len(presyn), size = 3):
#         presyn[idx] = np.append(presyn[idx], u)
    


#%%
import os
from snudda.utils.snudda_path import snudda_parse_path, get_snudda_data
from snudda import SnuddaLoad
from snudda.neurons.neuron_model_extended import NeuronModel
from snudda.simulate.nrn_simulator_parallel import NrnSimulatorParallel
from neuron import h 

from snudda.neurons.morphology_data import MorphologyData, SectionMetaData

network_path = '/Users/wst/Desktop/Karolinska/Simulation/Neuron/networks/sim_test_sparse_7/'
snudda_data  = '/Users/wst/Desktop/Karolinska/Simulation/Neuron'

original = SnuddaLoad(os.path.join(network_path, 'network-synapses.hdf5'))
original_neurons = original.data["neurons"]
original_synapses = original.data['synapses']
original_synapse_coords = original.data['synapse_coords']

original_id = 5
post_id_synapses = (original_synapses[:,1] == original_id).astype(bool)
pre_ids = list(set(original_synapses[post_id_synapses, 0]))
neurons_to_draw = [original_id] + pre_ids
neurons_to_draw = [original_id]
coords = original_synapse_coords[post_id_synapses]

ID = 5
neuron_rotation = original_neurons[0]["rotation"]
morph = snudda_parse_path(original_neurons[ID]["morphology"], snudda_data)
neuron_path = snudda_parse_path(original_neurons[ID]["neuron_path"],snudda_data)
param = os.path.join(neuron_path, "parameters.json")
mech = os.path.join(neuron_path, "mechanisms.json")
modulation = None

parameter_key = original_neurons[ID]["parameter_key"]
morphology_key = original_neurons[ID]["morphology_key"]
modulation_key = None
name = 'test'
neuron_0 = NeuronModel(param_file=param, morph_path=morph,mech_file=mech,
                                               cell_name=name,
                                               modulation_file=modulation,
                                               parameter_key=parameter_key,
                                               morphology_key=morphology_key,
                                               modulation_key=modulation_key)
neuron_0.instantiate(sim =NrnSimulatorParallel(cvode_active=False) )



#%%
all_points = []
for sec in neuron_0.icell.dend:
    print(sec)
    n_points = int(h.n3d(sec=sec))
    xyz = np.zeros((n_points, 3))
    for i in range(0, n_points):
        xyz[i, 0] = h.x3d(i, sec=sec)
        xyz[i, 1] = h.y3d(i, sec=sec)
        xyz[i, 2] = h.z3d(i, sec=sec)
    xyz = np.transpose(np.matmul(neuron_rotation, np.transpose(xyz)))
    print(xyz)
    
    all_points.append(np.array([np.array(p) for p in xyz]))
all_points = np.concatenate(all_points)
# np.savetxt("test_morph.csv", all_points, delimiter=",")    
    
    
#%%


start_row, end_row = find_next_synapse_group(original_synapses, neuron_id_on_node, next_row = 0)

sec_id = original_synapses[start_row:end_row, 9]
sec_x_list = original_synapses[start_row:end_row, 10] / 1000.0
voxel_coords = original_synapses[start_row:end_row, 2:5]
simulation_origo = original.data["simulation_origo"]
voxel_size = original.data["voxel_size"]
neuron_position = original.data["neurons"][ID]["position"]
neuron_rotation = original.data["neurons"][ID]["rotation"]

# Transform voxel coordinates to local neuron coordinates to match neuron
synapse_pos = (voxel_size * voxel_coords + simulation_origo - neuron_position) * 1e6

dend_sections = neuron_0.map_id_to_compartment(sec_id)
syn_pos_nrn = np.zeros((len(sec_id), 3))
old_sec = None
norm_arc_dist = None

for i, (sec, sec_x) in enumerate(zip(dend_sections, sec_x_list)):

    # If statement is just so we dont recalculate the norm_arc_dist every time
    if old_sec is None or not sec.same(old_sec):
        num_points = int(h.n3d(sec=sec))
        arc_dist = np.array([sec.arc3d(x) for x in range(0, num_points)])
        norm_arc_dist = arc_dist / arc_dist[-1]
        old_sec = sec

    # Find the closest point
    closest_idx = np.argmin(np.abs(norm_arc_dist - sec_x))

    syn_pos_nrn[i, 0] = h.x3d(closest_idx, sec=sec)
    syn_pos_nrn[i, 1] = h.y3d(closest_idx, sec=sec)
    syn_pos_nrn[i, 2] = h.z3d(closest_idx, sec=sec)

# We need to rotate the neuron to match the big simulation
# !!! OBS, this assumes that soma is in 0,0,0 local coordinates
syn_pos_nrn_rot = np.transpose(np.matmul(neuron_rotation,
                                         np.transpose(syn_pos_nrn)))






        # num_points = int(h.n3d(sec=sec))
        # arc_dist = np.array([sec.arc3d(x) for x in range(0, num_points)])
        # norm_arc_dist = arc_dist / arc_dist[-1]
        # old_sec = sec

    # Find the closest point
    # closest_idx = np.argmin(np.abs(norm_arc_dist - sec_x))

neuron_md = MorphologyData(swc_file=morph, parent_tree_info=None,
                                            snudda_data=snudda_data, lazy_loading=None)

# neuron_md.place(position=neuron_position, rotation=neuron_rotation, lazy=None)

#%%
import matplotlib.pyplot as plt


md_geo = neuron_md.geometry[421:422, :3] * 1e6
    
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], s = 5, c = 'black')

ax.scatter(md_geo[:,0], md_geo[:,1], md_geo[:,2], s = 100, c = 'green')
# ax.scatter(all_xyz[:,0], all_xyz[:,1], all_xyz[:,2], s = 45, c = 'purple')

# ax.scatter(md_geo[all_sec_id,0], md_geo[all_sec_id,1], md_geo[all_sec_id,2], s = 10, c = 'green')




dend_sections = neuron_0.map_id_to_compartment(all_sec_id)
syn_pos_nrn = np.zeros((len(all_sec_id), 3))
old_sec = None
norm_arc_dist = None

for i, (sec, sec_x) in enumerate(zip(dend_sections, all_sec_x)):

    # If statement is just so we dont recalculate the norm_arc_dist every time
    if old_sec is None or not sec.same(old_sec):
        num_points = int(h.n3d(sec=sec))
        arc_dist = np.array([sec.arc3d(x) for x in range(0, num_points)])
        norm_arc_dist = arc_dist / arc_dist[-1]
        old_sec = sec

    # Find the closest point
    closest_idx = np.argmin(np.abs(norm_arc_dist - sec_x))

    syn_pos_nrn[i, 0] = h.x3d(closest_idx, sec=sec)
    syn_pos_nrn[i, 1] = h.y3d(closest_idx, sec=sec)
    syn_pos_nrn[i, 2] = h.z3d(closest_idx, sec=sec)

# We need to rotate the neuron to match the big simulation
# !!! OBS, this assumes that soma is in 0,0,0 local coordinates
syn_pos_nrn_rot = np.transpose(np.matmul(neuron_rotation,
                                         np.transpose(syn_pos_nrn)))
ax.scatter(syn_pos_nrn_rot[:,0], syn_pos_nrn_rot[:,1], syn_pos_nrn_rot[:, 2], c = 'red')

for idx in [55]:
    
    sec_id = original_synapses[idx, 9]
    
    sec=neuron_0.icell.dend[sec_id]
    n_points = int(h.n3d(sec=sec))
    xyz = np.zeros((n_points, 3))
    for i in range(0, n_points):
        xyz[i, 0] = h.x3d(i, sec=sec)
        xyz[i, 1] = h.y3d(i, sec=sec)
        xyz[i, 2] = h.z3d(i, sec=sec)
    xyz = np.transpose(np.matmul(neuron_rotation, np.transpose(xyz)))
    
    
    
    
    num_points = int(h.n3d(sec=sec))
    arc_dist = np.array([sec.arc3d(x) for x in range(0, num_points)])
    norm_arc_dist = arc_dist / arc_dist[-1]
    old_sec = sec
    
    # Find the closest point
    closest_idx = np.argmin(np.abs(norm_arc_dist - sec_x_list[idx]))
    
    x = h.x3d(closest_idx, sec=sec)
    y= h.y3d(closest_idx, sec=sec)
    z = h.z3d(closest_idx, sec=sec)
    
    

    

    # ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s = 10)
    # ax.scatter(x,y,z, s = 100, c = 'red')
    # ax.scatter(synapse_pos[:,0], synapse_pos[:,1], synapse_pos[:, 2])
    # ax.scatter(syn_pos_nrn_rot[:,0], syn_pos_nrn_rot[:,1], syn_pos_nrn_rot[:, 2], c = 'red')
    
    ax.scatter(synapse_pos[idx,0], synapse_pos[idx,1], synapse_pos[idx, 2], c = 'purple', s = 100, marker = '^')
    ax.scatter(syn_pos_nrn_rot[idx,0], syn_pos_nrn_rot[idx,1], syn_pos_nrn_rot[idx, 2], c = 'black', s = 40, marker = 's')

plt.show()



#%%
from snudda.neurons import NeuronMorphologyExtended

nme = NeuronMorphologyExtended(name = 'test',
                                                                    position=neuron_position,  # This is set further down when using clone
                                                                    rotation=neuron_rotation,
                                                                    swc_filename=morph,
                                                                    snudda_data=snudda_data,
                            param_data=param, mech_filename=mech,
                                                                           parameter_key=parameter_key,
                                                                           morphology_key=morphology_key,
                                                                           modulation_key=modulation_key,
                                                                    load_morphology=True,
                                                                    virtual_neuron=False,
                                                                    verbose=False)



#%%
rng = np.random.default_rng()
section_data = neuron_md.section_data
geometry = neuron_md.geometry
soma_dist = geometry[:, 4]
parent_idx = section_data[:, 3]

keep_mask = section_data[:, 2] == 3
dend_idx = np.where(keep_mask)[0]



all_xyz, all_sec_id, all_sec_x, all_dist_to_soma = nme.dendrite_input_locations(rng = np.random.default_rng(), num_locations = 100, synapse_density_str = "1/(1+ exp((d-90e-6)/10e-6))")


all_xyz = (all_xyz -  neuron_position) * 1e6
synapse_density_str = "1/(1+ exp((d-90e-6)/10e-6))"

synapse_density, dend_idx = nme.get_weighted_synapse_density(synapse_density_str=synapse_density_str)

comp_len = soma_dist.copy()
comp_len[1:] -= soma_dist[parent_idx[1:]]
comp_synapse_density = (synapse_density + synapse_density[parent_idx]) / 2
expected_synapses = np.multiply(comp_len, comp_synapse_density)
expected_sum = np.sum(expected_synapses[dend_idx])



same_section = section_data[syn_idx, 0] == section_data[parent_idx[syn_idx], 0]
parent_sec_x = np.where(same_section, section_data[parent_idx[syn_idx], 1], 0.0)
sec_x = comp_x * section_data[syn_idx, 1] + (1-comp_x) * parent_sec_x
dist_to_soma = comp_x * geometry[syn_idx, 4] + (1-comp_x) * geometry[parent_idx[syn_idx], 4]






#%%

comp_x = rng.random(1)
syn_idx = 0
xyz = comp_x[:, None] * geometry[syn_idx, :3] + (1-comp_x[:, None]) * geometry[parent_idx[syn_idx], :3]
sec_id = section_data[syn_idx, 0]
sec_x_old = comp_x * section_data[syn_idx, 1] + (1-comp_x) * section_data[parent_idx[syn_idx], 1]


same_section = section_data[syn_idx, 0] == section_data[parent_idx[syn_idx], 0]

sec_x_start = np.where(same_section, section_data[parent_idx[syn_idx], 1], 0.0)
sec_x_end = section_data[syn_idx, 1]
sec_x = comp_x * sec_x_end + (1 - comp_x) * sec_x_start
print(sec_x)
print(sec_x_old)


#%%
neuron_id_on_node = np.zeros((100,), dtype=bool)
neuron_id_on_node[0] = True
neuron_id_on_node[2] = True
def find_next_synapse_group(synapses,neuron_id_on_node, next_row=0):

    """
    Synapses are sorted by destination neuron (and then source neuron), this method starts from next_row
    and find the next range of synapses that have the same source and destination.

    Args:
        next_row (int): Row in the synapse matrix to start from
    """


    try:
        num_syn_rows = synapses.shape[0]
    except:
        import traceback
        tstr = traceback.format_exc()

        # No more synapses to get
        return None

    # The synapse matrix is sorted on dest_id, ascending order
    # We also assume that self.neuron_id is sorted in ascending order

    start_row = None
    our_id = None
    not_our_id = None  # used in the loop below, despite what pycharm thinks

    while start_row is None:

        # What is the next destination ID
        next_id = synapses[next_row, 1]

        # Is the next ID ours?
        # TODO: This can be speed up by instead having a bool array with 1 if neuron is in self.neuron_id and 0
        #       otherwise. That would prevent us from having to search in self.neuron_id
        # if next_id in self.neuron_id: -- OLD if statement, replaced with bool lookup below
        
        
        if neuron_id_on_node[next_id]:
            start_row = next_row
            our_id = next_id
            continue
        else:
            not_our_id = next_id

        # This loop just skips all synapses targeting not_our_id so we then can check next id
        while (next_row < num_syn_rows and
               synapses[next_row, 1] == not_our_id):
            next_row += 1

        if next_row >= num_syn_rows:
            # No more synapses to get
            return None

    # Next find the last of the rows with this ID
    end_row = start_row

    while (end_row < num_syn_rows
           and synapses[end_row, 1] == our_id):
        end_row += 1

    return start_row, end_row





#%%%
syn_idx = rng.choice(a=dend_idx, size=unique_locations, replace=True,
                                     p=expected_synapses[dend_idx] / expected_sum)
#%%

soma_dist = geometry[:, 4]
parent_idx = section_data[:, 3]
# syn_idx = idx
num_locations = len(syn_idx)
comp_x = rng.random(num_locations)
sec_id = section_data[syn_idx, 0]




same_section = np.all(section_data[syn_idx][:, [0, 2]] == section_data[parent_idx[syn_idx]][:, [0, 2]], axis=1)

sec_x = np.where(same_section,
                  comp_x * section_data[syn_idx, 1] + (1-comp_x) * section_data[parent_idx[syn_idx], 1],
                  comp_x * section_data[syn_idx, 1])       


# mask = parent_idx[syn_idx] > 0
# sec_x = comp_x * section_data[syn_idx, 1]
# sec_x[mask] += (1 - comp_x[mask]) * section_data[parent_idx[syn_idx[mask]], 1]
          
xyz = comp_x[:, None] * geometry[syn_idx, :3] + (1-comp_x[:, None]) * geometry[parent_idx[syn_idx], :3]


for sec, sx, coords, comp, s_idx in zip(sec_id, sec_x, xyz, comp_x, syn_idx):
    sub = section_data[(section_data[:, 0] == sec)&(section_data[:, 2] == 3)]
    
    put_x = np.argmin(np.abs(sub[:,1] -sx))
    idx = np.where(np.all(section_data == sub[put_x], axis=1))[0]
    t = geometry[idx[0], :3]*1e6
    # print(geometry[idx[0], :3])1
    # print(coords)
    # print(np.linalg.norm(coords*1e6 - t))
    if np.linalg.norm(coords*1e6 - t) > 0.4:
        print(idx)
        print(s_idx)
        print(sec)
        print(sx)
        print(np.linalg.norm(coords*1e6 - t))
        print(coords*1e6)
        print(t)
        print(comp)
        print(section_data[idx, 3])
        print()
        keep_idx = s_idx
        break
        

#%%
syn_idx = [keep_idx]
num_locations = len(syn_idx)
for i in range(1000):
    comp_x = rng.random(num_locations)
    xyz = comp_x[:, None] * geometry[syn_idx, :3] + (1-comp_x[:, None]) * geometry[parent_idx[syn_idx], :3]
    print(xyz*1e6)
    if np.linalg.norm(xyz*1e6) < 50:
        print(comp_x)
        break
    
    
#%%

if parent_idx[syn_idx] > 0:
    sec_x = comp* section_data[syn_idx, 1] + (1-comp) * section_data[parent_idx[syn_idx], 1]
else:
    sec_x = comp* section_data[syn_idx, 1]

       

    
    
    
    
    