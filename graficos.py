from IPython.display import clear_output, display
import time
import matplotlib.pyplot as plt

############################################################
# PLOT SETTINGS
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator

# Font settings
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
# mpl.rc('text', usetex=True)

# Tick settings
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2

# Axis linewidth
mpl.rcParams['axes.linewidth'] = 2

# Tick direction and enabling ticks on all sides
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

# Function to apply custom tick locators and other settings to an Axes object
def apply_custom_settings(ax, log_scale_y=False):

    if log_scale_y:
        # Use LogLocator for the y-axis if it's in log scale
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    else:
        # Use AutoLocator for regular scales
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Apply the AutoLocator for the x-axis
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

############################################################

def plot_simulation(malla, h, u, N_elementos, time_step, number_of_t_step):

    clear_output(wait=True)  # Clear the output of the current cell    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot h in the top panel
    for i in range(N_elementos):
        ax1.plot(malla[i], h[i], linestyle='-', marker='o', markersize=5, linewidth=2)
        ax1.fill_between(malla[i], h[i], color='blue', alpha=1)
    ax1.axvspan(xmin=-1, xmax=0, ymin=0, ymax=1.10, facecolor='gray', alpha=1.)
    ax1.axvspan(xmin=10, xmax=11, ymin=0, ymax=1.10, facecolor='gray', alpha=1.)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax1.set_ylim(0.85, 1.14)
    ax1.set_ylabel(r'$h \, (m)$', fontsize=22)

    # Apply custom settings to ax1
    apply_custom_settings(ax1)

    # Plot u in the bottom panel
    for i in range(N_elementos):
        ax2.plot(malla[i], u[i], linestyle='-', marker='o', markersize=5, linewidth=2)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax2.set_ylim(-0.2, 0.2)
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_xlabel(r'$x \, (m)$', fontsize=22)
    ax2.set_ylabel(r'$u \, (m/s)$', fontsize=22)

    # Apply custom settings to ax2 (ensure this function is defined)
    apply_custom_settings(ax2)

    ax1.set_title(r'$t = {:.2f} \, s$'.format( (number_of_t_step+1) * time_step), fontsize=22)

    display(fig)
    plt.close(fig)
    time.sleep(0.1)  # Pause briefly to allow the display to update
