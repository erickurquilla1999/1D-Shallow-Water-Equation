import numpy as np
import utilities
import os
import evolve
import inputs

def write_data_file(element_n, nodes_coords,hgt,vel,vel_equal_hu,step):

    print(f'Writing step {step} | t = {step*inputs.t_step}')

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

def rk4_method(_h_, _u_, timestep, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_):

    # computing k1
    dh_dt, du_dt = evolve.compute_time_derivatives(_h_, _u_, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k1_h = dh_dt * timestep
    k1_u = du_dt * timestep

    # computing k2
    dh_dt, du_dt = evolve.compute_time_derivatives(_h_ + k1_h * 0.5, _u_ + k1_u * 0.5, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k2_h = dh_dt * timestep
    k2_u = du_dt * timestep

    # computing k3
    dh_dt, du_dt = evolve.compute_time_derivatives(_h_ + k2_h * 0.5, _u_ + k2_u * 0.5, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k3_h = dh_dt * timestep
    k3_u = du_dt * timestep

    # computing k4
    dh_dt, du_dt = evolve.compute_time_derivatives(_h_ + k3_h, _u_ + k3_u, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)
    k4_h = dh_dt * timestep
    k4_u = du_dt * timestep

    # computing values of h and u in next time step
    h_new = _h_ + (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6
    u_new = _u_ + (k1_u + 2 * k2_u + 2 * k2_u + k4_u) / 6

    return h_new, u_new

def euler_method(_h_, _u_, timestep, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_):

    # compute time derivatives of h and u
    dh_dt, du_dt = evolve.compute_time_derivatives(_h_, _u_, ele_nub_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, ele_len_, bas_vals_at_nod_, mass_matrix_inverse_)

    # evolving in time h and u with euler method
    h_new = _h_ + dh_dt * timestep
    u_new = _u_ + du_dt * timestep

    return h_new, u_new