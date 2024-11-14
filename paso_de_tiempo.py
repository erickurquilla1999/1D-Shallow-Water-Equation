import galerkin_discontinuo

def calcular_vector_residual(h__, u__, matriz_de_rigidez__):
    
    # computing stiffness vectors
    stiffness_vector_1_, stiffness_vector_2_ = galerkin_discontinuo.compute_stiffness_vectors(h__, u__, matriz_de_rigidez__)

    # computing numerical flux
    numerical_flux_vector_1_, numerical_flux_vector_2_ = galerkin_discontinuo.compute_numerical_flux_vectors(h__, u__)

    # computing residual vector
    residual_vector_1_ = stiffness_vector_1_ - numerical_flux_vector_1_
    residual_vector_2_ = stiffness_vector_2_ - numerical_flux_vector_2_

    return residual_vector_1_, residual_vector_2_