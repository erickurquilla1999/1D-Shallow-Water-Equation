import numpy as np
import utilities
import os
import evolve
import inputs

def euler_method(elmnt_numb,u1_,u2_,du1dt_, du2dt_,tstep,numb_t_step):
    
    print(f'Step: {numb_t_step}  |  t = {tstep*numb_t_step}')

    u1_nw=np.zeros((len(elmnt_numb),len(u1_[0])))
    u2_nw=np.zeros((len(elmnt_numb),len(u2_[0])))

    #looping over elements
    for n in elmnt_numb:

        for i in range(len(u1_[n])):
            u1_nw[n][i]=u1_[n][i]+du1dt_[n][i]*tstep

        for i in range(len(u2_[n])):
            u2_nw[n][i]=u2_[n][i]+du2dt_[n][i]*tstep

    return u1_nw, u2_nw

def write_data_file(element_n, nodes_coords,hgt,vel,vel_equal_hu,step):

    print(f'Writing step {step} ... ')

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

def rk4_method(elmnt_numb, u1,u2,f1,f2, basis_vals_at_nods, Nmatrix, Minv, timestep,numb_time_step):

    print(f'Step: {numb_time_step}  |  t = {timestep*numb_time_step}')

    def hf(elmnt_n,du1_dt_,du2_dt_,tstep):
        k_u1 = np.zeros((len(elmnt_numb),len(du1_dt_[0])))
        k_u2 = np.zeros((len(elmnt_numb),len(du2_dt_[0])))

        #looping over elements
        for n in elmnt_numb:

            for i in range(len(du1_dt_[n])):
                k_u1[n][i]=du1_dt_[n][i]*tstep

            for i in range(len(du2_dt_[n])):
                k_u2[n][i]=du2_dt_[n][i]*tstep
        return k_u1, k_u2

    # computing k1
    R_f_1, R_f_2 = evolve.compute_residual_vector(elmnt_numb,u1,u2,f1,f2,basis_vals_at_nods,Nmatrix)
    du1dt_, du2dt_ = evolve.compute_time_derivates(elmnt_numb,Minv, R_f_1, R_f_2)
    k1_u1_, k1_u2_ = hf(elmnt_numb,du1dt_,du2dt_,timestep)
    

    # computing k2
    u1_n = u1 + k1_u1_/np.array(2)
    u2_n = u2 + k1_u2_/np.array(2)
    f1_n = u2_n
    f2_n = np.where(u1_n == 0, 0, u2_n**2/u1_n + inputs.g * u1_n**2/np.array(2))

    R_f_1, R_f_2 = evolve.compute_residual_vector(elmnt_numb,u1_n,u2_n,f1_n,f2_n,basis_vals_at_nods,Nmatrix)
    du1dt_, du2dt_ = evolve.compute_time_derivates(elmnt_numb,Minv, R_f_1, R_f_2)
    k2_u1_, k2_u2_ = hf(elmnt_numb,du1dt_,du2dt_,timestep)

    # computing k3
    u1_n = u1 + k2_u1_/np.array(2)
    u2_n = u2 + k2_u2_/np.array(2)
    f1_n = u2_n
    f2_n = np.where(u1_n == 0, 0, u2_n**2/u1_n + inputs.g * u1_n**2/np.array(2))

    R_f_1, R_f_2 = evolve.compute_residual_vector(elmnt_numb,u1_n,u2_n,f1_n,f2_n,basis_vals_at_nods,Nmatrix)
    du1dt_, du2dt_ = evolve.compute_time_derivates(elmnt_numb,Minv, R_f_1, R_f_2)
    k3_u1_, k3_u2_ = hf(elmnt_numb,du1dt_,du2dt_,timestep)

    # computing k4
    u1_n = u1 + k3_u1_
    u2_n = u2 + k3_u2_
    f1_n = u2_n
    f2_n = np.where(u1_n == 0, 0, u2_n**2/u1_n + inputs.g * u1_n**2/np.array(2))

    R_f_1, R_f_2 = evolve.compute_residual_vector(elmnt_numb,u1_n,u2_n,f1_n,f2_n,basis_vals_at_nods,Nmatrix)
    du1dt_, du2dt_ = evolve.compute_time_derivates(elmnt_numb,Minv, R_f_1, R_f_2)
    k4_u1_, k4_u2_ = hf(elmnt_numb,du1dt_,du2dt_,timestep)

    # computing values of u_1 and u_2 in next time step
    u1_new = u1 + np.array(1/6)*(k1_u1_+np.array(2)*k2_u1_+np.array(2)*k2_u1_+k4_u1_) 
    u2_new = u2 + np.array(1/6)*(k1_u2_+np.array(2)*k2_u2_+np.array(2)*k2_u2_+k4_u2_) 

    return u1_new, u2_new