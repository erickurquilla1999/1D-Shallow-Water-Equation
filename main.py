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
basis_values_at_ref_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data,gauss_coords_in_elements,basis_values_at_gauss_coords,gauss_weights_in_elements,basis_derivative_values_at_gauss_coords = basis.generate_reference_space(inputs.N_elements,inputs.p_basis_order,inputs.out_x_points_per_element,inputs.n_gauss_poins,nodes_coord_ref_space)

# write initial conditions in output/step_0.h5 directory. f1=h and f2=hu
f1, f2 = initial_conditions.write_initial_conditions(element_number, nodes_coord, left_node_coords, right_node_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data)

# compute matrix M and return the inverse matrix of M
M_inverse = evolve.compute_M_matrix_inverse(element_number,basis_values_at_gauss_coords,gauss_weights_in_elements,left_node_coords, right_node_coords)

#mapping shallow-water equations to eq par_t u_i + par_x f_i = 0. u=(h,hu) and f=(hu,hu^2+gh^2/2). f1=h and f2=hu.
# setting the initil conditions to u and f components, u_i and f_i means u and f in component i 
u_1 = f1
u_2 = f2
f_1 = f2
f_2 = np.where(f1 == 0, 0, np.array(f2)**2 / f1 + inputs.g * np.array(f1)**2 / 2)

# evolving in time the PDE
for number_of_t_step in np.arange(inputs.n_steps):

    #computing residual vector
    R_f_1, R_f_2 = evolve.compute_residual_vector(element_number,u_1,u_2,f_1,f_2,basis_values_at_gauss_coords,basis_derivative_values_at_gauss_coords,left_node_coords, right_node_coords,gauss_weights_in_elements,basis_values_at_ref_coords)

    # compute time derivatives of u_1 and u_2
    du1_dt, du2_dt = evolve.compute_time_derivates(element_number,M_inverse, R_f_1, R_f_2)
