import numpy as np

import inputs
import grid_generation
import basis
import initial_conditions
import evolve 
import integrator

# creating mesh
element_number, left_node_coordinates, right_node_coordinates, nodes_coordinates_phys_space, nodes_coordinates_ref_space, element_lengths = grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# generating reference space information
gauss_weights, basis_values_at_gauss_quad, basis_values_time_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space(element_number,nodes_coordinates_phys_space,inputs.n_gauss_poins)

# generating initial conditions
h, u = initial_conditions.generate_initial_conditions(nodes_coordinates_phys_space)

# wrinting initial conditions file
integrator.write_data_file(element_number,nodes_coordinates_phys_space,h,u,False,0)

# compute matrix M and return the inverse matrix of M
M_inverse = evolve.compute_M_matrix_inverse(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad)

#mapping shallow-water equations to eq par_t u_i + par_x f_i = 0. u=(h,hu) and f=(hu,hu^2+gh^2/2). u_1=h and u_2=h*u
# setting the initil conditions to u and f components, u_i and f_i means u and f in component i 
u_1 = h
u_2 = h*u
f_1 = u_2
f_2 = np.where(u_1 == 0, 0, np.array(u_2)**2 / u_1 + inputs.g * np.array(u_1)**2 / 2)

# evolving in time the PDE
for number_of_t_step in np.arange(inputs.n_steps):

    #computing residual vector
    R_f_1, R_f_2 = evolve.compute_residual_vector(element_number,u_1,u_2,f_1,f_2,gauss_weights, basis_values_at_gauss_quad, basis_values_time_derivative_at_gauss_quad,element_lengths, basis_values_at_nodes)

    # compute time derivatives of u_1 and u_2
    du1_dt, du2_dt = evolve.compute_time_derivates(element_number,M_inverse, R_f_1, R_f_2)

    # evolving in time with euler method
    u_1_new, u_2_new = integrator.euler_method(element_number,u_1,u_2,du1_dt, du2_dt,inputs.t_step,number_of_t_step+1)

    # saving the data
    integrator.write_data_file(element_number,nodes_coordinates_phys_space,u_1_new,u_2_new,True,number_of_t_step+1)

    # saving new quantities to evolve next time step
    u_1 = u_1_new
    u_2 = u_2_new
    f_1 = u_2
    f_2 = np.where(f1 == 0, 0, np.array(u_2)**2 / u_1 + inputs.g * np.array(u_1)**2 / 2)