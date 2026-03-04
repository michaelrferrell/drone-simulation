# Attention default units are SI
import numpy as np
from conversions_constants import *

class Vehicle:
    def __init__(self, mass, inertia_tensor, r_cg, r_cp_ref):
        # Physical properties
        self.mass = mass
        
        # Process inertia tensor
        self.inertia_matrix = self.process_inertia(inertia_tensor)
        self.inertia_inv = np.linalg.inv(self.inertia_matrix) # Pre-calculate for Dynamics
        
        # Reference points (body frame)
        self.r_cg = np.array(r_cg)
        self.r_cp_ref = np.array(r_cp_ref) # Reference cp, actual cp will be updated based on center of attack

    # process_inertia
    # Accepts [Ixx, Iyy, Izz] OR a full 3x3 list/array and converts input into a 3x3 numpy array
    def process_inertia(self, input_val):
        data = np.array(input_val)
            
        if data.shape == (3,): # Diagonal
            return np.diag(data)
        elif data.shape == (3, 3): # Full tensor
            return data
        else:
            raise ValueError("Inertia must be a 3-element list or a 3x3 matrix.")
    
    # get_properties function
    # Returns the core properties for the Dynamics solver    
    def get_properties(self):
        return self.mass, self.inertia_matrix, self.inertia_inv, self.r_cg
    
    # update_mass function
    # Updates mass of vehicle
    def update_mass(self, delta_m):
        self.mass += delta_m         