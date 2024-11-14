import numpy as np
import os
import galerkin_discontinuo


def compute_time_derivatives(h__, u__, mass_matrix_inverse__, matriz_de_rigidez__):
    
    # computing stiffness vectors
    stiffness_vector_1_, stiffness_vector_2_ = galerkin_discontinuo.compute_stiffness_vectors(h__, u__, matriz_de_rigidez__)

    # computing numerical flux
    numerical_flux_vector_1_, numerical_flux_vector_2_ = galerkin_discontinuo.compute_numerical_flux_vectors(h__, u__)

    # computing residual vector
    residual_vector_1_ = stiffness_vector_1_ - numerical_flux_vector_1_
    residual_vector_2_ = stiffness_vector_2_ - numerical_flux_vector_2_

    # compute time derivatives of h and hu
    dh_dt_  = [mass_matrix_inverse__ @ res_vec_1_ for res_vec_1_ in residual_vector_1_]
    dhu_dt_ = [mass_matrix_inverse__ @ res_vec_2_ for res_vec_2_ in residual_vector_2_]

    # compute time derivatives of u
    du_dt_ = np.where( h__ == 0 , 0 , ( dhu_dt_ - u__ * dh_dt_ ) / h__ )

    return dh_dt_, du_dt_