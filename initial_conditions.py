import numpy as np
import initial_conditions
import utilities
import inputs
import basis
import os
import matplotlib.pyplot as plt

# initial conditions

# initial height as function of x
def initial_height(x):
    h=1+0.1*np.exp(-(x-5)**2)
    return h

# initial velocity as function of x
def velocity_initial(x):
    u=0
    return u

def write_initial_conditions(element_number, nodes_coords, left_node_coords, right_node_coords, ref_coords_to_save_data, basis_values_at_the_point_to_save_data):
    # Setting the initial height and velocity in each element node
    h_height = [np.array([initial_conditions.initial_height(x_node) for x_node in nodes]) for nodes in nodes_coords]
    u_velocity = [np.array([initial_conditions.velocity_initial(x_node) for x_node in nodes]) for nodes in nodes_coords]
    
    # Interpolating to save the output data / out means output data to plot
    x_out, h_out, u_out = [], [], []
    for n in element_number:
        x_loc = np.linspace(left_node_coords[n], right_node_coords[n], inputs.out_x_points_per_element + 1)
        x_out.append(x_loc)
        
        # interpolate the height and velocity from the base function to all the points
        basis_vals = np.array(basis_values_at_the_point_to_save_data[n])
        h_out.append(np.dot(basis_vals, h_height[n]))
        u_out.append(np.dot(basis_vals, u_velocity[n]))

    # Creating an output directory to save data
    os.makedirs('output', exist_ok=True)
    utilities.save_data_to_hdf5([element_number, nodes_coords, h_height, u_velocity, x_out, h_out, u_out],
                                ['element_number', 'nodes_coord', 'h_height', 'u_velocity', 'x_out', 'h_out', 'u_out'],
                                'output/step_0.h5')
    
    f1 = h_height
    f2 = np.multiply(h_height,u_velocity)

    return f1, f2

    # Testing the initial conditions scripts / see output/h_initial_conditions_test.pdf and see output/u_initial_conditions_test.pdf
    os.makedirs('tests', exist_ok=True)
    # plotting h from basis functions aproximation toguether with h given by the initial conditions equations 
    fig, ax = plt.subplots()
    for x, h in zip(x_out, h_out):
        ax.plot(x, h)
    x_vals = np.linspace(inputs.x_initial, inputs.x_final, inputs.out_x_points_per_element * inputs.N_elements)
    h_ini = np.array([initial_conditions.initial_height(x) for x in x_vals])
    ax.plot(x_vals, h_ini, c='black', linestyle='dotted', label='Real equation')
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r"$h$ (m)")
    ax.legend()
    fig.savefig('tests/h_initial_conditions_test.pdf', bbox_inches='tight')
    plt.clf()

    # plotting u from basis functions aproximation toguether with u given by the initial conditions equations 
    fig, ax = plt.subplots()
    for x, u in zip(x_out, u_out):
        ax.plot(x, u)
    u_ini = np.array([initial_conditions.velocity_initial(x) for x in x_vals])
    ax.plot(x_vals, u_ini, c='black', linestyle='dotted', label='Real equation')
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r"$u$ (m)")
    ax.legend()
    fig.savefig('tests/u_initial_conditions_test.pdf', bbox_inches='tight')
    plt.clf()