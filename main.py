import numpy as np
import grid_generation
import inputs
import os
import evolve 
import initial_conditions
import basis

# creating a generatedfiles directory to save generated files
try: os.makedirs('generatedfiles')
except: os.system('rm -r generatedfiles/*')
    
# creating mesh
grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# generationg reference space information
basis.generate_reference_space(inputs.p_basis_order,inputs.out_x_points_per_element,inputs.n_gauss_poins)

# write initial conditions in 
initial_conditions.write_initial_conditions()

# evolve the PDE
