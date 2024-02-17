import numpy as np
import inputs
import utilities
import os

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

def write_data_file(element_n, nodes_coords,hgt,vel,vel_equal_hu,step):

    if vel_equal_hu:
        vel=np.where(hgt == 0, 0, np.array(vel)/np.array(hgt))

    if step==0:
        # Check if the directory exists
        directory = 'output'
        if os.path.exists(directory):
            # If the directory exists, remove all files inside it
            file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
            for f in file_list:
                os.remove(f)
        else:
            # If the directory does not exist, create it
            os.makedirs(directory)

    utilities.save_data_to_hdf5([element_n, nodes_coords, hgt, vel],
                                ['element_number', 'nodes_coordinates', 'height', 'velocity'],
                                'output/step_'+str(step)+'.h5')