import numpy as np
import os
import discontinuos_galerkin


def compute_time_derivatives(h__, u__, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, gau_wei, _malla_, mass_matrix_inverse__):
    
    # computing stiffness vectors
    stiffness_vector_1_, stiffness_vector_2_ = discontinuos_galerkin.compute_stiffness_vectors(_malla_, gau_wei, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, h__, u__)

    # computing numerical flux
    numerical_flux_vector_1_, numerical_flux_vector_2_ = discontinuos_galerkin.compute_numerical_flux_vectors(h__, u__)

    # computing residual vector
    residual_vector_1_ = stiffness_vector_1_ - numerical_flux_vector_1_
    residual_vector_2_ = stiffness_vector_2_ - numerical_flux_vector_2_

    # compute time derivatives of h and hu
    dh_dt_  = [mass_matrix_inverse__ @ res_vec_1_ for res_vec_1_ in residual_vector_1_]
    dhu_dt_ = [mass_matrix_inverse__ @ res_vec_2_ for res_vec_2_ in residual_vector_2_]

    # compute time derivatives of u
    du_dt_ = np.where( h__ == 0 , 0 , ( dhu_dt_ - u__ * dh_dt_ ) / h__ )

    return dh_dt_, du_dt_

def rk4_method(_h_, _u_, timestep, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, malla__, mass_matrix_inverse_):

    # computing k1
    dh_dt, du_dt = compute_time_derivatives(_h_, _u_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, malla__, mass_matrix_inverse_)
    k1_h = dh_dt * timestep
    k1_u = du_dt * timestep

    # computing k2
    dh_dt, du_dt = compute_time_derivatives(_h_ + k1_h * 0.5, _u_ + k1_u * 0.5, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, malla__, mass_matrix_inverse_)
    k2_h = dh_dt * timestep
    k2_u = du_dt * timestep

    # computing k3
    dh_dt, du_dt = compute_time_derivatives(_h_ + k2_h * 0.5, _u_ + k2_u * 0.5, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, malla__, mass_matrix_inverse_)
    k3_h = dh_dt * timestep
    k3_u = du_dt * timestep

    # computing k4
    dh_dt, du_dt = compute_time_derivatives(_h_ + k3_h, _u_ + k3_u, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, malla__, mass_matrix_inverse_)
    k4_h = dh_dt * timestep
    k4_u = du_dt * timestep

    # computing values of h and u in next time step
    h_new = _h_ + (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6
    u_new = _u_ + (k1_u + 2 * k2_u + 2 * k2_u + k4_u) / 6

    return h_new, u_new

def euler_method(_h_, _u_, timestep, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, malla__, mass_matrix_inverse_):

    # compute time derivatives of h and u
    dh_dt, du_dt = compute_time_derivatives(_h_, _u_, bas_vals_at_gau_quad_, bas_vals_x_der_at_gau_quad_, gau_wei_, malla__, mass_matrix_inverse_)

    # evolving in time h and u with euler method
    h_new = _h_ + dh_dt * timestep
    u_new = _u_ + du_dt * timestep

    return h_new, u_new