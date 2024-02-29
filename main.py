import numpy as np

import inputs
import grid_generation
import basis
import initial_conditions
import evolve 
import integrator
import plots
import test

# running some test
test.test_lagrange_basis()
test.test_lagrange_basis_derivative()
test.test_integration()
test.test_mass_matrix_1()

# creating mesh
element_number, left_node_coordinates, right_node_coordinates, nodes_coordinates_phys_space, nodes_coordinates_ref_space, element_lengths = grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# generating reference space information
gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space(element_number,nodes_coordinates_phys_space,inputs.n_gauss_poins,left_node_coordinates, right_node_coordinates)

# generating initial conditions
h, u = initial_conditions.generate_initial_conditions(nodes_coordinates_phys_space)

# wrinting initial conditions file
integrator.write_data_file(element_number,nodes_coordinates_phys_space,h,u,False,0,0)

# compute mass matrix 1 : M_ij = integral phi_i(x) phi_j(x) dx and return the inverse
mass_matrix_1_inverse = evolve.compute_mass_matrix_1_inverse(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad)

# count time
time = 0

# evolving in time the PDE
for number_of_t_step in np.arange(inputs.n_steps):

    # If true using euler method otherwise use RK4
    if inputs.evolution_method==0:
        
        #################################################################################################
        # solving for height h

        # computing stiffness vector 1 : integral ( d_dx phi_i(x) ) h u dx and return the inverse
        stiffness_vector_1 = evolve.compute_stiffness_vector_1(element_number, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, gauss_weights, element_lengths, h, u)

        # computing numerical flux 1 : phi_i(b) f_rho(b) - phi_i(a) f_rho(a)
        numerical_flux_vector_1 = evolve.compute_numerical_flux_vector_1(element_number, basis_values_at_nodes, h, u)

        # compute residual vector 1
        residual_vector_1 = stiffness_vector_1 - numerical_flux_vector_1

        # compute time derivative of h
        dh_dt = [mass_mat_inv @ res_vec_1 for mass_mat_inv, res_vec_1 in zip(mass_matrix_1_inverse, residual_vector_1)]

        #################################################################################################
        # solving for velocity u

        # compute mass matrix 2 : M_ij = integral phi_i(x) phi_j(x) h dx and return the inverse
        mass_matrix_2_inverse = evolve.compute_mass_matrix_2_inverse(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, h)

        # compute mass vector 2 complement: integral phi_i(x) ( d_dt h ) u dx and return the inverse
        mass_vector_2_complement = evolve.compute_mass_vector_2_complement(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, dh_dt, u)

        # compute stiffness vector 2 : integral ( d_dt phi_i(x) ) ( h u^2 + g h^2 / 2) dx and return the inverse
        stiffness_vector_2 = evolve.compute_stiffness_vector_2(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, h, u)

        # computing numerical flux 2 : phi_i(b) f_rho(b) - phi_i(a) f_rho(a)
        numerical_flux_vector_2 = evolve.compute_numerical_flux_vector_2(element_number, basis_values_at_nodes, h, u)

        # computing residual vector 2
        residual_vector_2 = stiffness_vector_2 - numerical_flux_vector_2 - mass_vector_2_complement

        # compute time derivatives of u
        du_dt = [mass_mat_inv @ res_vec_2 for mass_mat_inv, res_vec_2 in zip(mass_matrix_2_inverse, residual_vector_2)]

        #################################################################################################

        # compute next time steps
        time_step = 0.1 * np.min([1/np.max(np.abs(dh_dt)) , 1/np.max(np.abs(du_dt)), inputs.t_step])

        # evolving in time with euler method
        h = h + dh_dt * np.array(time_step)
        u = u + du_dt * np.array(time_step)

        # count time
        time += time_step

    else:
        u_1_new, u_2_new = integrator.rk4_method(element_number,u_1,u_2,f_1,f_2,basis_values_at_nodes,N,M_inverse,np.array(inputs.t_step),number_of_t_step+1)

    # saving the data
    if (number_of_t_step+1) % inputs.plot_every_steps == 0:
        integrator.write_data_file(element_number,nodes_coordinates_phys_space,h,u,False,number_of_t_step+1, time)

# plotting data
plots.plotting()

print(f'Done')