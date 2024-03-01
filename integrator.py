import numpy as np
import utilities
import os
import evolve
import inputs

def write_data_file(element_n, nodes_coords,hgt,vel,vel_equal_hu,step,time_):

    print(f'Writing step {step} | t = {time_} ... ')

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

    utilities.save_data_to_hdf5([element_n, nodes_coords, hgt, vel, time_],
                                ['element_number', 'nodes_coordinates', 'height', 'velocity','time'],
                                'output/step_'+str(step)+'.h5')

def rk4_method( _h_, _u_, timestep, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_):

    def compute_time_derivatives(h__, u__, ele_nub, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, gau_wei, ele_len, bas_vals_at_nod, mass_matrix_inverse):

        stiffness_vector_1 = evolve.compute_stiffness_vector_1(ele_nub, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, gau_wei, ele_len, h__, u__)
        numerical_flux_vector_1 = evolve.compute_numerical_flux_vector_1(ele_nub, bas_vals_at_nod, h__, u__)
        residual_vector_1 = stiffness_vector_1 - numerical_flux_vector_1
        dh_dt_ = [mass_mat_inv @ res_vec_1 for mass_mat_inv, res_vec_1 in zip(mass_matrix_inverse, residual_vector_1)]

        stiffness_vector_2 = evolve.compute_stiffness_vector_2(ele_nub, ele_len, gau_wei, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, h__, u__)
        numerical_flux_vector_2 = evolve.compute_numerical_flux_vector_2(ele_nub, bas_vals_at_nod, h__, u__)
        residual_vector_2 = stiffness_vector_2 - numerical_flux_vector_2
        dhu_dt_ = [mass_mat_inv @ res_vec_2 for mass_mat_inv, res_vec_2 in zip(mass_matrix_inverse, residual_vector_2)]

        return dh_dt_, dhu_dt_

    # computing k1
    dh_dt, dhu_dt = compute_time_derivatives(_h_, _u_, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k1_h  = dh_dt  * timestep
    k1_hu = dhu_dt * timestep

    # computing k2
    dh_dt, dhu_dt = compute_time_derivatives(_h_ + 0.5 * k1_h, _u_ + 0.5 * k1_hu / _h_, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k2_h  = dh_dt  * timestep
    k2_hu = dhu_dt * timestep

    # computing k3
    dh_dt, dhu_dt = compute_time_derivatives(_h_ + 0.5 * k2_h, _u_ + 0.5 * k2_hu / _h_, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k3_h  = dh_dt  * timestep
    k3_hu = dhu_dt * timestep

    # computing k4
    dh_dt, dhu_dt = compute_time_derivatives(_h_ + 0.5 * k3_h, _u_ + k3_hu / _h_, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k4_h  = dh_dt  * timestep
    k4_hu = dhu_dt * timestep

    # computing values of u_1 and u_2 in next time step
    h_new = _h_ + np.array(1/6)*(k1_h + np.array(2) * k2_h + np.array(2) * k3_h + k4_h) 
    u_new = _u_ + np.array(1/6)*(k1_hu + np.array(2) * k2_hu + np.array(2) * k2_hu + k4_hu) / _h_

    return h_new, u_new