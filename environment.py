# Attention default units are SI
import numpy as np
from ambiance import Atmosphere
from conversions_constants import *

class Environment:
    def __init__(self, air_density=RHO_SEA_LEVEL, ambient_pressure=P_SEA_LEVEL, ambient_temperature=T_SEA_LEVEL, gravity=STANDARD_GRAVITY):
        # Initial defaults (sea level)
        self.rho_sl = air_density
        self.p_sl = ambient_pressure       
        self.t_sl = ambient_temperature    
        self.g0 = gravity     
        
        # Current state values
        self.rho = air_density
        self.p = ambient_pressure
        self.t = ambient_temperature
        self.g = gravity             

    # update function
    # Updates environment properties based on current altitude (upper limit currently is 81,020 m)
    def update(self, altitude):
        atm = Atmosphere(altitude)

        self.rho = atm.density[0]
        self.p = atm.pressure[0]
        self.t = atm.temperature[0]
        self.g = atm.grav_accel[0]

        return self.rho, self.p, self.t, self.g