import numpy as np

# initial conditions

# initial height as function of x
def initial_height(x):
    h=1+0.1*np.exp(-(x-5)**2)
    return h

# initial velocity as function of x
def velocity_initial(x):
    u=0
    return u
