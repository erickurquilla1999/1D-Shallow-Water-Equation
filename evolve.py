import numpy as np
import initial_conditions
import utilities
import inputs
import basis
import os


def compute_M_matrix_inverse(element_number,basis_values_at_gauss_coords,gauss_weights_in_elements,left_node_coords, right_node_coords):
    #Lopp over all element
    M = []
    M_inverse = []

    for n in element_number:
        phi = np.array(basis_values_at_gauss_coords[n])
        weights = gauss_weights_in_elements[n]
        delta_x = right_node_coords[n] - left_node_coords[n]
        
        # Compute M for the current element
        M_ele_n = 0.5 * delta_x * np.dot(phi.T * weights, phi)
        # Append M to the list
        M.append(M_ele_n)

        # Compute the inverse of M for the current element
        M_inv_ele_n = np.linalg.inv(M_ele_n)

        # Append the inverse of M to the list
        M_inverse.append(M_inv_ele_n)

    return M_inverse

def compute_residual_vector(element_number,u_1,u_2,f_1,f_2,basis_values_at_gauss_coords,basis_derivative_values_at_gauss_coords,left_node_coords, right_node_coords,gauss_weights_in_elements,basis_values_at_ref_coords):

    R_f_1=[]
    R_f_2=[]

    for n in element_number:
        #interpolationg f_1 to gauss cuadrature
        f_1_gauss=np.dot(basis_values_at_gauss_coords[n],f_1[n])
        f_2_gauss=np.dot(basis_values_at_gauss_coords[n],f_2[n])

        dphi_dx=np.array(basis_derivative_values_at_gauss_coords[n])

        # resicual vector 1 
        R_1_f_1 = 0.5*(right_node_coords[n]-left_node_coords[n])*np.dot(dphi_dx.T,f_1_gauss)
        R_1_f_2 = 0.5*(right_node_coords[n]-left_node_coords[n])*np.dot(dphi_dx.T,f_2_gauss)

        basis_at_initial_ele_node = basis_values_at_ref_coords[n][0]
        basis_at_final_ele_node = basis_values_at_ref_coords[n][len(basis_values_at_ref_coords[n])-1]

         # computing Roe flux
        roe_flux_1_left=0
        roe_flux_1_right=0
        roe_flux_2_left=0
        roe_flux_2_right=0

        if n==0:
            
            roe_flux_1_left=0
            roe_flux_2_left=inputs.g*u_1[n][n]**2/2

            Jacobian_right = -inputs.g * 0.5 * ( u_1[n][len(u_1[n])-1] + u_1[n+1][0]) + ( ( f_1[n][len(f_1[n])-1] + f_1[n+1][0] ) / ( u_1[n][len(u_1[n])-1] + u_1[n+1][0] ) ) **2
            
            roe_flux_1_right = 0.5 * ( ( f_1[n][len(f_1[n])-1] + f_1[n+1][0] ) ) - 0.5 * Jacobian_right * ( u_1[n][len(u_1[n])-1] + u_1[n+1][0] )
            roe_flux_2_right = 0.5 * ( ( f_2[n][len(f_2[n])-1] + f_2[n+1][0] ) ) - 0.5 * Jacobian_right * ( u_2[n][len(u_2[n])-1] + u_2[n+1][0] )
        
        elif n==len(element_number)-1:
        
            Jacobian_left = -inputs.g * 0.5 * ( u_1[n-1][len(u_1[n])-1] + u_1[n][0]) + ( ( f_1[n-1][len(f_1[n])-1] + f_1[n][0] ) / ( u_1[n-1][len(u_1[n])-1] + u_1[n][0] ) ) **2
            
            roe_flux_1_left = 0.5 * ( ( f_1[n-1][len(f_1[n])-1] + f_1[n][0] ) ) - 0.5 * Jacobian_left * ( u_1[n-1][len(u_1[n])-1] + u_1[n][0] )
            roe_flux_2_left = 0.5 * ( ( f_2[n-1][len(f_2[n])-1] + f_2[n][0] ) ) - 0.5 * Jacobian_left * ( u_2[n-1][len(u_2[n])-1] + u_2[n][0] )

            roe_flux_1_right=0
            roe_flux_2_right=inputs.g*u_1[n][len(f_1[n])-1]**2/2

        else:

            Jacobian_left = -inputs.g * 0.5 * ( u_1[n-1][len(u_1[n])-1] + u_1[n][0]) + ( ( f_1[n-1][len(f_1[n])-1] + f_1[n][0] ) / ( u_1[n-1][len(u_1[n])-1] + u_1[n][0] ) ) **2
            
            roe_flux_1_left = 0.5 * ( ( f_1[n-1][len(f_1[n])-1] + f_1[n][0] ) ) - 0.5 * Jacobian_left * ( u_1[n-1][len(u_1[n])-1] + u_1[n][0] )
            roe_flux_2_left = 0.5 * ( ( f_2[n-1][len(f_2[n])-1] + f_2[n][0] ) ) - 0.5 * Jacobian_left * ( u_2[n-1][len(u_2[n])-1] + u_2[n][0] )
            
            Jacobian_right = -inputs.g * 0.5 * ( u_1[n][len(u_1[n])-1] + u_1[n+1][0]) + ( ( f_1[n][len(f_1[n])-1] + f_1[n+1][0] ) / ( u_1[n][len(u_1[n])-1] + u_1[n+1][0] ) ) **2
            
            roe_flux_1_right = 0.5 * ( ( f_1[n][len(f_1[n])-1] + f_1[n+1][0] ) ) - 0.5 * Jacobian_right * ( u_1[n][len(u_1[n])-1] + u_1[n+1][0] )
            roe_flux_2_right = 0.5 * ( ( f_2[n][len(f_2[n])-1] + f_2[n+1][0] ) ) - 0.5 * Jacobian_right * ( u_2[n][len(u_2[n])-1] + u_2[n+1][0] )

        # resicual vector 2
        R_2_f_1 = -(np.array(basis_at_final_ele_node) * np.array(roe_flux_1_right) - np.array(basis_at_initial_ele_node) * np.array(roe_flux_1_left))
        R_2_f_2 = -(np.array(basis_at_final_ele_node) * np.array(roe_flux_2_right) - np.array(basis_at_initial_ele_node) * np.array(roe_flux_2_right))

        # adding residual vector 1 and 2
        R_f_1.append(R_1_f_1+R_2_f_1)
        R_f_2.append(R_1_f_2+R_2_f_2)
    
    return R_f_1, R_f_2

def compute_time_derivates(element_number,M_inverse, R_f_1, R_f_2 ):

    du1dt=[]
    du2dt=[]

    #Lopp over all element
    for n in element_number:
        du1dt.append(np.dot(M_inverse[n],R_f_1[n]))    
        du2dt.append(np.dot(M_inverse[n],R_f_2[n]))    

    return du1dt, du2dt




