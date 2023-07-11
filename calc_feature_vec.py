import numpy as np
def feature_extraction_terminal_distance(char1:np.array,char2:np.array,rel_coord1:np.array,rel_coord2:np.array)->np.array:
    """
    A function to calculate feature vector of two molecules, unsing principal information and relative coordinates of two related molecules.
    Params:
        char1: characteristic vectors of molecule 1, uses following input
            [principle axies,principal center]. e.g., [[[1,0,0],[0,1,0],[0,0,1]],[0,0,0]]
        char2: characteristic vectors of molecule 2, uses the same format above.
        rel_coord1: list of relative coordinates for atoms in molecule1, under its own principal axies.
        rel_coord2: list of relative coordinates for atoms in molecule2, under its own principal axies.

    """
    
    axis1 = char1[0]
    axis2 = char2[0]
    rel_dis = char2[1] - char1[1]
    
    """
    This section is hard-wired atom idx for the determination of terminal, and central fused rings. The idx listed here was extracted from the reference y6_ref.sdf file provided in the repository.
    If one uses different ordering of atoms, he should change the section to corresponding atoms.
    """
    terminal1_ring_idx = [54,55,56,57,58,60]
    terminal2_ring_idx = [2,3,5,6,7,8]
    core_ring_idx = [37,38,42,43,81,91]
    
    #Extract the relative coordinates of terminal and core ring atoms
    terminal1_1_coords=rel_coord1[terminal1_ring_idx]
    terminal1_2_coords=rel_coord1[terminal2_ring_idx]
    ring1_coords = rel_coord1[core_ring_idx]
    
    terminal2_1_coords=rel_coord2[terminal1_ring_idx]
    terminal2_2_coords=rel_coord2[terminal2_ring_idx]
    ring2_coords = rel_coord2[core_ring_idx]
    
    #generate terminal and core center coordinate
    terminal1_1 = np.mean(terminal1_1_coords,axis=0)
    terminal1_2 = np.mean(terminal1_2_coords,axis=0)
    core1 = np.mean(ring1_coords,axis=0)
    
    terminal2_1 = np.mean(terminal2_1_coords,axis=0)
    terminal2_2 = np.mean(terminal2_2_coords,axis=0)
    core2 = np.mean(ring2_coords,axis=0)
    
    #translate into laboratory axis
    terminal_1_1_actual = terminal1_1[0]*char1[0][0] + terminal1_1[1]*char1[0][1] + terminal1_1[2]*char1[0][2] + char1[1]
    terminal_1_2_actual = terminal1_2[0]*char1[0][0] + terminal1_2[1]*char1[0][1] + terminal1_2[2]*char1[0][2] + char1[1]
    core_1_actual = core1[0]*char1[0][0] + core1[1]*char1[0][1] + core1[2]*char1[0][2] + char1[1]
    
    terminal_2_1_actual = terminal2_1[0]*char2[0][0] + terminal2_1[1]*char2[0][1] + terminal2_1[2]*char2[0][2] + char2[1]
    terminal_2_2_actual = terminal2_2[0]*char2[0][0] + terminal2_2[1]*char2[0][1] + terminal2_2[2]*char2[0][2] + char2[1]
    core_2_actual = core2[0]*char2[0][0] + core2[1]*char2[0][1] + core2[2]*char2[0][2] + char2[1]
    
    #calculate positional vector
    terminal_1_to_1 = terminal_1_1_actual - terminal_2_1_actual
    terminal_1_to_2 = terminal_1_1_actual - terminal_2_2_actual
    terminal_2_to_1 = terminal_1_2_actual - terminal_2_1_actual
    terminal_2_to_2 = terminal_1_2_actual - terminal_2_2_actual
    terminal_11_to_core2 = terminal_1_1_actual - core_2_actual
    terminal_12_to_core2 = terminal_1_2_actual - core_2_actual
    terminal_21_to_core1 = terminal_2_1_actual - core_1_actual
    terminal_22_to_core1 = terminal_2_2_actual - core_2_actual
    core1_to_core_2 = core_1_actual - core_2_actual
    
    #Keep only the shortest distance considering the symmetry
    a_a_vector = [terminal_1_to_1,terminal_1_to_2,terminal_2_to_1,terminal_2_to_2]
    a_a_prime_vector = [terminal_11_to_core2,terminal_12_to_core2,terminal_21_to_core1,terminal_22_to_core1]
    aa_distance = [np.linalg.norm(x) for x in a_a_vector]
    aap_distance = [np.linalg.norm(x) for x in a_a_prime_vector]
    a_a_vector = a_a_vector[aa_distance.index(min(aa_distance))]
    a_a_prime_vector = a_a_prime_vector[aap_distance.index(min(aap_distance))]
    a_prime_a_prime_vector = core1_to_core_2
    
    #calculate relative position vector projection
    a_a_proj = [np.dot(axis1[0],a_a_vector),np.dot(axis1[1],a_a_vector),np.dot(axis1[2],a_a_vector)]
    a_ap_proj = [np.dot(axis1[0],a_a_prime_vector),np.dot(axis1[1],a_a_prime_vector),np.dot(axis1[2],a_a_prime_vector)]
    ap_ap_proj = [np.dot(axis1[0],a_prime_a_prime_vector),np.dot(axis1[1],a_prime_a_prime_vector),np.dot(axis1[2],a_prime_a_prime_vector)]
    
    #generate the rotary matrix between molecule 1 and molecule 2
    rotate_mat = [np.dot(axis1[0],axis2[0]),np.dot(axis1[1],axis2[1]),np.dot(axis1[2],axis2[2])]
    
    #calculate projection of relative position vector of two molecules
    rel_projection = [np.dot(axis1[0],rel_dis),np.dot(axis1[1],rel_dis),np.dot(axis1[2],rel_dis)]

    #summarize the final feature vector
    feature_vec = a_a_proj + a_ap_proj + ap_ap_proj + rotate_mat + rel_projection

    return feature_vec