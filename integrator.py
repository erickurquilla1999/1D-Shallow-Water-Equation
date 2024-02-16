import numpy as np
import inputs
import utilities
import os
import matplotlib.pyplot as plt

def euler_method(element_number,u_1,u_2,du1_dt, du2_dt):

    #time step
    t_step=inputs.t_limit/inputs.n_steps # s 
    
    u1_nw=np.zeros((len(element_number),len(u_1[0])))
    u2_nw=np.zeros((len(element_number),len(u_1[0])))

    #looping over elements
    for n in element_number:

        for i in range(len(u_1[n])):
            u1_nw[n][i]=u_1[n][i]+du1_dt[n][1]*t_step

        for i in range(len(u_2[n])):
            u2_nw[n][i]=u_2[n][i]+du2_dt[n][1]*t_step

    return u1_nw, u2_nw

def write_data_file(number_of_t_step,u_1_new, u_2_new,element_number, nodes_coords, left_node_coords, right_node_coords, basis_values_at_the_point_to_save_data):
    # Computing height and velocity
    h_height = u_1_new
    u_velocity = np.where(u_1_new == 0, 0, np.array(u_2_new)/np.array(u_1_new))

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
    utilities.save_data_to_hdf5([element_number, nodes_coords, h_height, u_velocity, x_out, h_out, u_out],
                                ['element_number', 'nodes_coord', 'h_height', 'u_velocity', 'x_out', 'h_out', 'u_out'],
                                'output/step_'+str(number_of_t_step+1)+'.h5')

    # plotting h from basis functions aproximation toguether with h given by the initial conditions equations 
    if (inputs.n_steps/(number_of_t_step+1)) % 1==0:
        fig, ax = plt.subplots()
        for x, h in zip(x_out, h_out):
            ax.plot(x, h)
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r"$h$ (m)")
        # ax.legend()
        fig.savefig('tests/h_step_'+str(number_of_t_step+1)+'.pdf', bbox_inches='tight')
        plt.clf()

        # plotting u from basis functions aproximation toguether with u given by the initial conditions equations 
        fig, ax = plt.subplots()
        for x, u in zip(x_out, u_out):
            ax.plot(x, u)
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r"$u$ (m)")
        # ax.legend()
        fig.savefig('tests/u_step_'+str(number_of_t_step+1)+'.pdf', bbox_inches='tight')
        plt.clf()