import numpy as np
import utilities
import os

def euler_method(elmnt_numb,u1_,u2_,du1dt_, du2dt_,tstep,numb_t_step):
    
    print(f'\nStep: {numb_t_step}  |  t = {tstep*numb_t_step}\n')

    u1_nw=np.zeros((len(elmnt_numb),len(u1_[0])))
    u2_nw=np.zeros((len(elmnt_numb),len(u2_[0])))

    #looping over elements
    for n in elmnt_numb:

        for i in range(len(u1_[n])):
            u1_nw[n][i]=u1_[n][i]+du1dt_[n][1]*tstep

        for i in range(len(u2_[n])):
            u2_nw[n][i]=u2_[n][i]+du2dt_[n][1]*tstep

    return u1_nw, u2_nw

def write_data_file(element_n, nodes_coords,hgt,vel,vel_equal_hu,step):

    print(f'\nWriting step {step} ... \n')

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