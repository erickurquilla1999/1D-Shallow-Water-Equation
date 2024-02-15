import numpy as np
import grid_generation
import inputs
import os
import evolve 

# creating a generatedfiles directory to save generated files
try: os.makedirs('generatedfiles')
except: os.system('rm -r generatedfiles/*')
    
# creating mesh
grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# write initial conditions in 
evolve.write_initial_conditions()

# evolve the PDE
