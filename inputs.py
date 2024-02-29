# input parameters

# Physical domain 
x_initial = 0 # (m) initial domain coordinate
x_final = 10 # (m) final domain coordinate

# DG method
N_elements= 10 # number of elements
p_basis_order = 2 # lagrange basis order

# simulation time
n_steps = 100 # number of time steps
t_step = 0.0001 # (s) simulation time limit

# Gauss cuadrature
n_gauss_poins = 20 

# constant
g = 9.8 # m/s^2

# plotting setting
plot_every_steps = 5

# PDE evolution method # 0: euler, 1: rk4
evolution_method = 0