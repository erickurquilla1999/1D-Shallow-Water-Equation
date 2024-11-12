import numpy as np

def lagrange_basis_derivative(nodes, i, x):
    """
    Compute the derivative of the Lagrange basis function for a given node and value of x.

    Parameters:
        nodes (array-like): Array of nodes in the element.
        i (int): Index of the node for which to compute the derivative.
        x (float): Value of x for which to compute the derivative.

    Returns:
        float: The derivative of the Lagrange basis function at the given node and value of x.
    """
    basis_derivative = 0
    for j in range(len(nodes)):
        if j != i:
            pc = 1
            for k in range(len(nodes)):
                if k != j and k != i:
                    pc *= (x-nodes[k])/(nodes[i]-nodes[k])
            basis_derivative += pc/(nodes[i]-nodes[j])
    return basis_derivative

def generate_reference_space(malla_, cuadratura_de_gauss_, polinomios_de_lagrange_):

    # guardando la cuadratura de Gauss en el espacio físico
    cuadratura_de_gauss_espacio_fisico = 0.5 * ( malla_[0][-1] - malla_[0][0] ) * cuadratura_de_gauss_ + 0.5 * ( malla_[0][-1] + malla_[0][0])

    # evaluando la función base en los puntos de cuadratura de Gauss
    polinomios_de_lagrange_en_cuadratura_de_gauss_ = np.array([[polinomios_de_lagrange_(malla_[0], i, x) for x in cuadratura_de_gauss_espacio_fisico] for i in range(len(malla_[0]))])

    # evaluando la derivada en x de la función base evaluada en los puntos de cuadratura de Gauss
    derivada_x_polinomios_de_lagrange_en_cuadratura_de_gauss_ = np.array([[lagrange_basis_derivative(malla_[0], i, x) for x in cuadratura_de_gauss_espacio_fisico] for i in range(len(malla_[0]))])

    return polinomios_de_lagrange_en_cuadratura_de_gauss_, derivada_x_polinomios_de_lagrange_en_cuadratura_de_gauss_