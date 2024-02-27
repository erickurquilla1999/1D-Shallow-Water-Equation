import numpy as np
import basis
import matplotlib.pyplot as plt
import evolve 

def basis_and_its_derivative():
    
    domain = np.linspace(-1,1,100)
    nodes = [-1,0,1]

    lagrange_basis=[]
    for i in range(len(nodes)):
        lagrange_basis_i=[]
        for x in domain:
            lagrange_basis_i.append(basis.lagrange_basis(nodes,i,x))
        lagrange_basis.append(lagrange_basis_i)

    fig, ax = plt.subplots()
    ax.plot(domain,lagrange_basis[0],c='b')
    ax.plot(domain,lagrange_basis[1],c='r')
    ax.plot(domain,lagrange_basis[2],c='g')
    ax.plot(domain,0.5*(domain**2-domain),c='black',linestyle='dotted')
    ax.plot(domain,1-domain**2,c='black',linestyle='dotted')
    ax.plot(domain,0.5*(domain**2+domain),c='black',linestyle='dotted')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\phi$')
    fig.savefig('plots/lagrange_basis.pdf',bbox_inches='tight')
    plt.close(fig) 

    lagrange_basis_der=[]
    for i in range(len(nodes)):
        lagrange_basis_der_i=[]
        for x in domain:
            lagrange_basis_der_i.append(basis.lagrange_basis_derivative(nodes,i,x))
        lagrange_basis_der.append(lagrange_basis_der_i)

    fig, ax = plt.subplots()
    ax.plot(domain,lagrange_basis_der[0],c='b')
    ax.plot(domain,lagrange_basis_der[1],c='r')
    ax.plot(domain,lagrange_basis_der[2],c='g')
    ax.plot(domain,0.5*(2*domain-np.array(1)),c='black',linestyle='dotted')
    ax.plot(domain,-2*domain,c='black',linestyle='dotted')
    ax.plot(domain,0.5*(2*domain+np.array(1)),c='black',linestyle='dotted')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$d\phi/dx$')
    fig.savefig('plots/lagrange_basis_derivative.pdf',bbox_inches='tight')
    plt.close(fig) 

def integration():

    gauss_weights, basis_values_at_gauss_quad, basis_values_time_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space([0],[[-1,0,1]],10,[-1],[1])

    x= np.array([-1,0,1])
    funtion_at_nodes = (x**2)*np.array(3)
    f_xn = basis_values_at_gauss_quad[0] @ funtion_at_nodes
    integral = 0.5 * ( 1 - ( -1 ) ) * np.sum(gauss_weights[0]*f_xn)
    print(f'\nintegrating 3*x**2 from -1 to 1')
    print(f'gauss integration: {integral}')
    print(f'real integration: {2}')
    
    x= np.array([-1,0,1])
    funtion_at_nodes = x
    f_xn = basis_values_at_gauss_quad[0] @ funtion_at_nodes
    integral = 0.5 * ( 1 - ( -1 ) ) * np.sum(gauss_weights[0]*f_xn*np.array(basis_values_time_derivative_at_gauss_quad[0])[:,0])
    print(f'\nintegrating x * dphi_1/dx from -1 to 1')
    print(f'gauss integration: {integral}')
    print(f'real integration: {0.66666666666666666}')

def M_matrix():
    
    gauss_weights, basis_values_at_gauss_quad, basis_values_time_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space([0],[[0,0.5,1]],30,[0],[1])
    M_inverse = evolve.compute_M_matrix_inverse([0], [1], gauss_weights, basis_values_at_gauss_quad)
    print(f'M matrix test')
    print(f'Value in the code {np.linalg.inv(M_inverse)}')
    print(f'Theory [0.1333333333,0.0666666666,-0.0333333333],[ - , 0.5333333333,0.0666666666],[ - , - , 0.13333333]')
    print(f'Unit matrix {M_inverse @ np.linalg.inv(M_inverse)}')

def N_matrix():
    
    gauss_weights, basis_values_at_gauss_quad, basis_values_time_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space([0],[[0,0.5,1]],30,[0],[1])
    N_matr = evolve.compute_N_matrix([0], basis_values_at_gauss_quad, basis_values_time_derivative_at_gauss_quad, gauss_weights,[1])
    print(f'N matrix test')
    print(f'Value in the code {N_matr}')
    print(f'Theory [-0.5,-0.66666666,0.1666666],[0.666666666,0,-0.66666666],[ -1.66666 , 0.66666666 , 0.5]')


