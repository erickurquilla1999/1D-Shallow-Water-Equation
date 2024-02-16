import numpy as np
import grid_generation
import inputs
import os
import evolve 
import initial_conditions
import basis

# creating a generatedfiles directory to save generated files
os.makedirs('generatedfiles', exist_ok=True)

# creating mesh
element_number, nodes_coord,  left_node_coords, right_node_coords, nodes_coord_ref_space = grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# generationg reference space information
basis_values_at_ref_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data,gauss_coords_in_elements,basis_values_at_gauss_coords,gauss_weights_in_elements = basis.generate_reference_space(inputs.N_elements,inputs.p_basis_order,inputs.out_x_points_per_element,inputs.n_gauss_poins,nodes_coord_ref_space)

# write initial conditions in output/step_0.h5 directory
f1, f2 = initial_conditions.write_initial_conditions(element_number, nodes_coord, left_node_coords, right_node_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data)

# compute matrix M and return the inverse matrix of M
M_inverse = evolve.compute_M_matrix_inverse(element_number,basis_values_at_gauss_coords,gauss_weights_in_elements,left_node_coords, right_node_coords)

# compute time derivatives of f1 and f2
evolve.compute_time_derivates(f1, f2, element_number)
