import numpy as np
import basis
import matplotlib.pyplot as plt
import evolve
import random

def test_lagrange_basis():
    # Test parameters
    num_nodes = 8
    nodes = np.random.uniform(0, 10, size=num_nodes)  # Generate random Lagrange nodes between 0 and 10
    i = 4  # Random index of the Lagrange node for which to compute the basis function
    x = np.random.uniform(0, 10)  # Random point at which to evaluate the basis function

    # Manually compute the expected value of the Lagrange basis function
    expected_value = (x - nodes[0]) * (x - nodes[1]) * (x - nodes[2]) * (x - nodes[3]) * (x - nodes[5]) * (x - nodes[6]) * (x - nodes[7]) / (
    (nodes[i] - nodes[0]) * (nodes[i] - nodes[1]) * (nodes[i] - nodes[2]) * (nodes[i] - nodes[3]) * (nodes[i] - nodes[5]) * (nodes[i] - nodes[6]) * (nodes[i] - nodes[7]))

    # Call the lagrange_basis function
    result = basis.lagrange_basis(nodes, i, x)

    # Check if the result matches the expected value
    if not np.isclose(result, expected_value):
        print(f"test_lagrange_basis() failed: Expected {expected_value}, but got {result}")

def test_lagrange_basis_derivative():
    # Test parameters
    num_nodes = 8
    nodes = np.random.uniform(0, 10, size=num_nodes)  # Generate random Lagrange nodes between 0 and 10
    i = 4  # Random index of the Lagrange node for which to compute the basis function derivative
    x = np.random.uniform(0, 10)  # Random point at which to evaluate the basis function derivative

    # Manually compute the expected value of the Lagrange basis function derivative
    expected_value = ((x-nodes[1])*(x-nodes[2])*(x-nodes[3])*(x-nodes[5])*(x-nodes[6])*(x-nodes[7])+(x-nodes[0])*(x-nodes[2])*(x-nodes[3])*(x-nodes[5])*(x-nodes[6])*(x-nodes[7])+(x-nodes[0])*(x-nodes[1])*(x-nodes[3])*(x-nodes[5])*(x-nodes[6])*(x-nodes[7])+(x-nodes[0])*(x-nodes[1])*(x-nodes[2])*(x-nodes[5])*(x-nodes[6])*(x-nodes[7])+(x-nodes[0])*(x-nodes[1])*(x-nodes[2])*(x-nodes[3])*(x-nodes[6])*(x-nodes[7])+(x-nodes[0])*(x-nodes[1])*(x-nodes[2])*(x-nodes[3])*(x-nodes[5])*(x-nodes[7])+(x-nodes[0])*(x-nodes[1])*(x-nodes[2])*(x-nodes[3])*(x-nodes[5])*(x-nodes[6])) / ((nodes[i] - nodes[0]) * (nodes[i] - nodes[1]) * (nodes[i] - nodes[2]) * (nodes[i] - nodes[3]) * (nodes[i] - nodes[5]) * (nodes[i] - nodes[6]) * (nodes[i] - nodes[7]))

    # Call the lagrange_basis derivative function
    result = basis.lagrange_basis_derivative(nodes, i, x)

    # Check if the result matches the expected value
    if not np.isclose(result, expected_value):
        print(f"test_lagrange_basis_derivative() failed: Expected {expected_value}, but got {result}")

def test_integration():

    n_gauss_quad_pnts = 10
    a = 0
    b = np.pi/2

    num_nodes = random.randint(2,4)   

    random_numbers = a + np.sort(np.random.rand(num_nodes-2)) * (b - a)
    nodes = np.concatenate(([a], random_numbers, [b]))
    
    gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space([0],[nodes],n_gauss_quad_pnts,[nodes[0]],[nodes[-1]])
                                                                                                                                            #    (element_nuber, nodes, gauss_quad_points, node_left, node_right)
    funtion_at_nodes = np.cos(nodes)
    funtion_at_quadrature_points = basis_values_at_gauss_quad[0] @ funtion_at_nodes
    result = 0.5 * ( np.pi/2 - 0 ) * np.sum(gauss_weights[0]*funtion_at_quadrature_points)

    # computing expected value
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss_quad_pnts)
    scaled_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    scaled_weights = 0.5 * (b - a) * weights
    expected_value = np.sum(scaled_weights * np.cos(scaled_nodes))

    if not np.isclose(result, expected_value):
        print(f"test_integration() failed: Expected {expected_value}, but got {result}")

def test_mass_matrix():

    num_nodes = 8
    n_gauss_quad_pnts = 10
    a = 0
    b = 1
    i = 4

    random_numbers = a + np.sort(np.random.rand(num_nodes-2)) * (b - a)
    nodes = np.concatenate(([a], random_numbers, [b]))

    gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space([0],[nodes],n_gauss_quad_pnts,[nodes[0]],[nodes[-1]])
    M_inverse = evolve.compute_mass_matrix_inverse([0], [b-a], gauss_weights, basis_values_at_gauss_quad)
                                                # (element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad)
    result = np.linalg.inv(M_inverse[0])[4,4]

    # computing expected value
    nodes_gauss, weights = np.polynomial.legendre.leggauss(n_gauss_quad_pnts)
    scaled_nodes = 0.5 * (b - a) * nodes_gauss + 0.5 * (b + a)
    scaled_weights = 0.5 * (b - a) * weights
    x = scaled_nodes
    lagrange_base_4_in_quad_points = (x - nodes[0]) * (x - nodes[1]) * (x - nodes[2]) * (x - nodes[3]) * (x - nodes[5]) * (x - nodes[6]) * (x - nodes[7]) / ((nodes[i] - nodes[0]) * (nodes[i] - nodes[1]) * (nodes[i] - nodes[2]) * (nodes[i] - nodes[3]) * (nodes[i] - nodes[5]) * (nodes[i] - nodes[6]) * (nodes[i] - nodes[7]))

    expected_value = np.sum(scaled_weights * lagrange_base_4_in_quad_points*lagrange_base_4_in_quad_points)
    
    if not np.isclose(result, expected_value):
        print(f"test_mass_matrix() failed: Expected {expected_value}, but got {result}")