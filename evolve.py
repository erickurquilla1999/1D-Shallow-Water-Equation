import numpy as np
import initial_conditions
import utilities
import inputs
import basis


# writing initial condition file: step_0.h5

def write_initial_conditions():

    #reading mesh file
    element_number = utilities.load_data_from_hdf5('element_number', 'generatedfiles/grid.h5')
    nodes_coords = utilities.load_data_from_hdf5('nodes_coords', 'generatedfiles/grid.h5')

    print(nodes_coords)

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

    x_out = np.linspace(inputs.x_initial, inputs.x_final, inputs.out_number_x_points + 1)
    h_out=[]
    u_out=[]

    for x in x_out:
        h_loc = 0
        for n in element_number:
            h_loc += basis.lagrange_interpolation(nodes_coords[n], h_height[n], x)
        h_out.append(h_loc)
    
    for x in x_out:
        u_loc = 0
        for n in element_number:
            u_loc += basis.lagrange_interpolation(nodes_coords[n], u_velocity[n], x)
        u_out.append(u_loc)
    

    import matplotlib.pyplot as plt     
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
    import matplotlib as mpl
    import matplotlib.ticker

    mpl.rcParams['font.size'] = 22
    mpl.rcParams['font.family'] = 'serif'
    mpl.rc('text', usetex=True)
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.major.pad'] = 8
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['axes.linewidth'] = 2

    fig, ax = plt.subplots()

    ax.plot(x_out,h_out)

    h_theory=[]
    
    for x in x_out:
        h_theory.append(initial_conditions.initial_height(x))

    ax.plot(x_out,h_theory)

    fig.savefig('t_step_0.pdf',bbox_inches='tight')
    plt.show()
    plt.clf()
