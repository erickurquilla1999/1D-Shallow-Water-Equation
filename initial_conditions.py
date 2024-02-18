import numpy as np

# initial conditions

def generate_initial_conditions(nodes_coords_all_elem):

    print('\nSaving initial conditions ... \n')

    # initial height as function of x
    def initial_height(x):
        h=1+0.1*np.exp(-(x-5)**2)
        return h

    # initial velocity as function of x
    def velocity_initial(x):
        u=0
        return u

    # Setting the initial height and velocity in each element node
    h_height = [np.array([initial_height(x_n) for x_n in nodes_in_elem]) for nodes_in_elem in nodes_coords_all_elem]
    u_velocity = [np.array([velocity_initial(x_n) for x_n in nodes_in_elem]) for nodes_in_elem in nodes_coords_all_elem]

    return h_height, u_velocity

