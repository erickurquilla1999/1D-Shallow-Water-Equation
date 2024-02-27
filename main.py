import numpy as np

import inputs
import grid_generation
import basis
import initial_conditions
import evolve 
import integrator
import plots
import test

# creating mesh
element_number, left_node_coordinates, right_node_coordinates, nodes_coordinates_phys_space, nodes_coordinates_ref_space, element_lengths = grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# generating reference space information
gauss_weights, basis_values_at_gauss_quad, basis_values_time_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space(element_number,nodes_coordinates_phys_space,inputs.n_gauss_poins,left_node_coordinates, right_node_coordinates)

# generating initial conditions
h, u = initial_conditions.generate_initial_conditions(nodes_coordinates_phys_space)

# wrinting initial conditions file
integrator.write_data_file(element_number,nodes_coordinates_phys_space,h,u,False,0)

# compute mass matrix M_ij = integral phi_i(x) phi_j(x) dx and return the inverse matrix of M_ij
mass_matrix_inverse = evolve.compute_mass_matrix_inverse(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad)

# compute stiffness matrix S_ij = integral dphi_i_dx(x) phi_j(x) dx
stiffness_matrix = evolve.compute_stiffness_matrix(element_number, basis_values_at_gauss_quad, basis_values_time_derivative_at_gauss_quad, gauss_weights,element_lengths)

#mapping shallow-water equations to eq par_t u_i + par_x f_i = 0. u=(h,hu) and f=(hu,hu^2+gh^2/2). u_1=h and u_2=h*u
# setting the initil conditions to u and f components, u_i and f_i means u and f in component i 
u_1 = h
u_2 = h * u
f_1 = h
f_2 = h * u**2 + inputs.g * h**2 / 2

# evolving in time the PDE
for number_of_t_step in np.arange(inputs.n_steps):

    # If true using euler method otherwise use RK4
    if inputs.evolution_method==0:
        
        # computing stiffness vector
        stiffness_vector_1 = [stif_mtx @ u_1_ for stif_mtx, u_1_ in zip(stiffness_matrix, u_1)]
        stiffness_vector_2 = [stif_mtx @ u_2_ for stif_mtx, u_2_ in zip(stiffness_matrix, u_2)]

        # computing numerical flux
        numerical_flux_vector_1, numerical_flux_vector_2 = evolve.compute_numerical_flux_vector(element_number,u_1,u_2,f_1,f_2,basis_values_at_nodes)

        # computing residual vector
        residual_vector_1 = stiffness_vector_1 + numerical_flux_vector_1
        residual_vector_2 = stiffness_vector_2 + numerical_flux_vector_2

        # compute time derivatives of u_1 and u_2
        du1_dt = [mass_mat_inv @ res_vec_1 for mass_mat_inv, res_vec_1 in zip(mass_matrix_inverse, residual_vector_1)]
        du2_dt = [mass_mat_inv @ res_vec_2 for mass_mat_inv, res_vec_2 in zip(mass_matrix_inverse, residual_vector_2)]

        # evolving in time with euler method
        u_1_new = u_1 + du1_dt * inputs.t_step
        u_2_new = u_2 + du2_dt * inputs.t_step

    else:
        u_1_new, u_2_new = integrator.rk4_method(element_number,u_1,u_2,f_1,f_2,basis_values_at_nodes,N,M_inverse,np.array(inputs.t_step),number_of_t_step+1)

    # saving the data
    if (number_of_t_step+1) % inputs.write_every_steps == 0:
        integrator.write_data_file(element_number,nodes_coordinates_phys_space,u_1_new,u_2_new,True,number_of_t_step+1)

    # saving new quantities to evolve next time step
    u_1 = u_1_new
    u_2 = u_2_new
    f_1 = u_2
    f_2 = np.where(u_1 == 0, 0, np.array(u_2)**2 / u_1 + inputs.g * np.array(u_1)**2 / 2)

# plotting data
plots.plotting()

print(f'Done')

# running some test
test.basis_and_its_derivative()
test.integration()
test.M_matrix()
test.N_matrix()
