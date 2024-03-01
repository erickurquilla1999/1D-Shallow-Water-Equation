import numpy as np
import utilities
import os
import evolve
import inputs

def write_data_file(element_n, nodes_coords,hgt,vel,vel_equal_hu,step):

    print(f'Writing step {step} | t = {step*inputs.t_step} ... ')

    if vel_equal_hu:
        vel=np.where(hgt == 0, 0, np.array(vel)/np.array(hgt))

    if step==0:
        # Check if the directory exists
        directory = 'output'
        if os.path.exists(directory):
            # If the directory exists, remove all files inside it
            file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
            for f in file_list:
                os.remove(f)
        else:
            # If the directory does not exist, create it
            os.makedirs(directory)

    utilities.save_data_to_hdf5([element_n, nodes_coords, hgt, vel,step*inputs.t_step],
                                ['element_number', 'nodes_coordinates', 'height', 'velocity','time'],
                                'output/step_'+str(step)+'.h5')

def rk4_method( _h_, _u_, timestep, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_):

    def compute_time_derivatives(h__, u__, ele_nub, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, gau_wei, ele_len, bas_vals_at_nod, mass_matrix_inverse__):
        
        # computing stiffness vectors
        stiffness_vector_1_, stiffness_vector_2_ = evolve.compute_stiffness_vectors(ele_nub, ele_len, gau_wei, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, h__, u__)

        # computing numerical flux
        numerical_flux_vector_1_, numerical_flux_vector_2_ = evolve.compute_numerical_flux_vectors(ele_nub, bas_vals_at_nod, h__, u__)

        # computing residual vector
        residual_vector_1_ = stiffness_vector_1_ - numerical_flux_vector_1_
        residual_vector_2_ = stiffness_vector_2_ - numerical_flux_vector_2_

        # compute time derivatives of u_1 and u_2
        dh_dt_ = [mass_mat_inv__ @ res_vec_1_ for mass_mat_inv__, res_vec_1_ in zip(mass_matrix_inverse__, residual_vector_1_)]
        dhu_dt_ = [mass_mat_inv__ @ res_vec_2_ for mass_mat_inv__, res_vec_2_ in zip(mass_matrix_inverse__, residual_vector_2_)]

        # compute time derivatives of u
        du_dt_ = np.where( h__ == 0 , 0 , ( dhu_dt_ - u__ * dh_dt_ ) / h__ )

        return dh_dt_, du_dt_

    # computing k1
    dh_dt, du_dt = compute_time_derivatives(_h_, _u_, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k1_h = dh_dt * timestep
    k1_u = du_dt * timestep

    # computing k2
    dh_dt, du_dt = compute_time_derivatives(_h_ + k1_h * 0.5, _u_ + k1_u * 0.5, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k2_h = dh_dt * timestep
    k2_u = du_dt * timestep

    # computing k3
    dh_dt, du_dt = compute_time_derivatives(_h_ + k2_h * 0.5, _u_ + k2_u * 0.5, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k3_h = dh_dt * timestep
    k3_u = du_dt * timestep

    # computing k4
    dh_dt, du_dt = compute_time_derivatives(_h_ + k3_h, _u_ + k3_u, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k4_h = dh_dt * timestep
    k4_u = du_dt * timestep

    # computing values of u_1 and u_2 in next time step
    h_new = _h_ + (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6
    u_new = _u_ + (k1_u + 2 * k2_u + 2 * k2_u + k4_u) / 6

    return h_new, u_new

def euler_method( h__, u__, timestep, ele_nub, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, gau_wei, ele_len, bas_vals_at_nod, mass_matrix_inverse__):

    # computing stiffness vectors
    stiffness_vector_1_, stiffness_vector_2_ = evolve.compute_stiffness_vectors(ele_nub, ele_len, gau_wei, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, h__, u__)

    # computing numerical flux
    numerical_flux_vector_1_, numerical_flux_vector_2_ = evolve.compute_numerical_flux_vectors(ele_nub, bas_vals_at_nod, h__, u__)

    # computing residual vector
    residual_vector_1_ = stiffness_vector_1_ - numerical_flux_vector_1_
    residual_vector_2_ = stiffness_vector_2_ - numerical_flux_vector_2_

    # compute time derivatives of u_1 and u_2
    dh_dt = [mass_mat_inv__ @ res_vec_1_ for mass_mat_inv__, res_vec_1_ in zip(mass_matrix_inverse__, residual_vector_1_)]
    dhu_dt = [mass_mat_inv__ @ res_vec_2_ for mass_mat_inv__, res_vec_2_ in zip(mass_matrix_inverse__, residual_vector_2_)]

    # compute time derivatives of u
    du_dt = np.where( h__ == 0 , 0 , ( dhu_dt - u__ * dh_dt ) / h__ )

    # evolving in time with euler method
    h_new = h__ + dh_dt * timestep
    u_new = u__ + du_dt * timestep

    return h_new, u_new


