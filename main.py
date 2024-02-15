import numpy as np
import grid_generation
import inputs

# creating mesh
grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements)