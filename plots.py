import matplotlib.pyplot as plt
import os
import utilities

def plotting():

    print('Plotting data ... ')
        
    # Check if the directory exists
    directory = 'plots'
    if os.path.exists(directory):
        # If the directory exists, remove all files inside it
        file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
        for f in file_list:
            os.remove(f)
    else:
        # If the directory does not exist, create it
        os.makedirs(directory)

    direc = "output/"
    files = os.listdir(direc)

    # Custom sorting function
    def sort_key(s):
        return int(s.split('_')[1].split('.')[0])

    # Sort the list based on the numbers after the underscore
    files = sorted(files, key=sort_key)

    for file in files:
    
        print(f'Plotting {file}')
         
        x = utilities.load_data_from_hdf5('nodes_coordinates','output/'+file)
        velocity = utilities.load_data_from_hdf5('velocity','output/'+file)
        height = utilities.load_data_from_hdf5('height','output/'+file)

        fig, ax = plt.subplots()
        for i in range(len(x)):
            ax.plot(x[i],height[i])
            # ax.scatter(x[i],height[i])
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$h$ (m)')
        ax.set_ylim(0.5,1.5)
        fig.savefig('plots/h_'+file[0:-3]+'.pdf',bbox_inches='tight')
        plt.close(fig) 

        fig, ax = plt.subplots()
        for i in range(len(x)):
            ax.plot(x[i],velocity[i])
            ax.scatter(x[i],velocity[i])
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$u$ (m/s)')
        fig.savefig('plots/u_'+file[0:-3]+'.pdf',bbox_inches='tight')
        plt.close(fig) 
