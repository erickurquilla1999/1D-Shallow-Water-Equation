import numpy as np
import initial_conditions
import utilities
import inputs
import basis
import os


# writing initial condition file: step_0.h5

def write_initial_conditions():

    #reading mesh file
    element_number = utilities.load_data_from_hdf5('element_number', 'generatedfiles/grid.h5')
    nodes_coords = utilities.load_data_from_hdf5('nodes_coords', 'generatedfiles/grid.h5')

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
    
    left_node_coords = utilities.load_data_from_hdf5('left_node_coords', 'generatedfiles/grid.h5')
    right_node_coords = utilities.load_data_from_hdf5('right_node_coords', 'generatedfiles/grid.h5')

    x_out=[]
    h_out=[]
    u_out=[]

    for n in element_number:
        
        x_loc = np.linspace(left_node_coords[n], right_node_coords[n], inputs.out_x_points_per_element + 1)
        x_out.append(x_loc)
        h_loc=[]
        u_loc=[]

        for x in x_loc:
            h_loc.append(basis.lagrange_interpolation(nodes_coords[n], h_height[n], x))
        h_out.append(h_loc)
        
        for x in x_loc:
            u_loc.append(basis.lagrange_interpolation(nodes_coords[n], u_velocity[n], x))
        u_out.append(u_loc)

    # creating a output directory to save data
    try: os.makedirs('output')
    except: os.system('rm -r output/*')

    utilities.save_data_to_hdf5([element_number,nodes_coords,h_height,u_velocity,x_out,h_out,u_out], ['element_number','nodes_coord','h_height','u_velocity','x_out','h_out','u_out'], 'output/step_1.h5')

    # import matplotlib.pyplot as plt     
    # fig, ax = plt.subplots()
    # for n in element_number:
    #     ax.plot(x_out[n],h_out[n])

    # h_theory=[]    
    # for x in np.linspace(inputs.x_initial,inputs.x_final,inputs.out_x_points_per_element*inputs.N_elements):
    #     h_theory.append(initial_conditions.initial_height(x))
    # ax.plot(np.linspace(inputs.x_initial,inputs.x_final,inputs.out_x_points_per_element*inputs.N_elements),h_theory,c='black',linestyle='dotted')

    # fig.savefig('t_step_0.pdf',bbox_inches='tight')
    # plt.show()
    # plt.clf()

