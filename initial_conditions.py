import numpy as np
import initial_conditions
import utilities
import inputs
import basis
import os

# initial conditions

# initial height as function of x
def initial_height(x):
    h=1+0.1*np.exp(-(x-5)**2)
    return h

# initial velocity as function of x
def velocity_initial(x):
    u=0
    return u
    
# writing initial condition file: step_0.h5

def write_initial_conditions(element_number,nodes_coords,left_node_coords,right_node_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data):

    # setting the initial height and velocity in each element node
    h_height=[]
    u_velocity=[]

    for n in element_number:
        h_ele=[]
        u_ele=[]
        for x_node in nodes_coords[n]:
            h_ele.append(initial_conditions.initial_height(x_node))
            u_ele.append(initial_conditions.velocity_initial(x_node))
        h_height.append(h_ele)
        u_velocity.append(u_ele)
    
    # inpolating to save the output data
    x_out=[]
    h_out=[]
    u_out=[]

    for n in element_number:
        
        x_loc = np.linspace(left_node_coords[n], right_node_coords[n], inputs.out_x_points_per_element + 1)
        x_out.append(x_loc)

        h_loc=[]
        for e in range(inputs.out_x_points_per_element+1):
            h_loc.append(np.dot(basis_values_at_the_point_to_save_data[n][e], h_height[n]))
        h_out.append(h_loc)

        u_loc=[]
        for e in range(inputs.out_x_points_per_element+1):
            u_loc.append(np.dot(basis_values_at_the_point_to_save_data[n][e], u_velocity[n]))
        u_out.append(u_loc)

    # creating a output directory to save data
    try: os.makedirs('output')
    except: os.system('rm -r output/*')

    utilities.save_data_to_hdf5([element_number,nodes_coords,h_height,u_velocity,x_out,h_out,u_out], ['element_number','nodes_coord','h_height','u_velocity','x_out','h_out','u_out'], 'output/step_1.h5')

    # testing the initial conditions scripts / see output/h_initial_conditions_test.pdf and see output/u_initial_conditions_test.pdf

    import matplotlib.pyplot as plt  

    fig, ax = plt.subplots()
    for n in element_number:
        ax.plot(x_out[n],h_out[n])
    h_ini=[]    
    for x in np.linspace(inputs.x_initial,inputs.x_final,inputs.out_x_points_per_element*inputs.N_elements):
        h_ini.append(initial_conditions.initial_height(x))
    ax.plot(np.linspace(inputs.x_initial,inputs.x_final,inputs.out_x_points_per_element*inputs.N_elements),h_ini,c='black',linestyle='dotted',label='Real equation')
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r"$h$ (m)")
    ax.legend()
    fig.savefig('output/h_initial_conditions_test.pdf',bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots()
    for n in element_number:
        ax.plot(x_out[n],u_out[n])
    u_ini=[]
    for x in np.linspace(inputs.x_initial,inputs.x_final,inputs.out_x_points_per_element*inputs.N_elements):
        u_ini.append(initial_conditions.velocity_initial(x))
    ax.plot(np.linspace(inputs.x_initial,inputs.x_final,inputs.out_x_points_per_element*inputs.N_elements),u_ini,c='black',linestyle='dotted',label='Real equation')
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r"$u$ (m)")
    ax.legend()
    fig.savefig('output/u_initial_conditions_test.pdf',bbox_inches='tight')
    plt.clf()