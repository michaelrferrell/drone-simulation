# Attention default units are SI
import numpy as np
from conversions_constants import *

import numpy as np

class Dynamics:
    def __init__(self):
        pass

    # compute_derivatives function
    # Computes rates of change for vehicle state
    def compute_derivatives(self, state, vehicle, force_body, torque_body, environment):
        """      
        Equations used:
        1. r_ddot = (1/m) * (R * T - m*g)
        2. omega_dot = J_inv * (-omega x J*omega + tau)
        3. q_dot = (1/2) * q ⊗ [0, omega]
        """
        
        # Translational acceleration (inertial frame)
        R = state.get_rotation_matrix() 
        force_inertial = R @ force_body
        accel_linear = (force_inertial / vehicle.mass) - np.array([0, 0, environment.g])

        # Angular acceleration (body frame)
        omega = state.omega
        J = vehicle.inertia_matrix
        J_inv = vehicle.inertia_inv
        
        gyro_term = np.cross(-omega, J @ omega)
        accel_angular = J_inv @ (gyro_term + torque_body)

        # Orientation derivative
        quat = state.quaternion
        p, q, r = omega
        
        # Mapping omega to the quaternion rate of change
        # quat indices: 0=w, 1=x, 2=y, 3=z
        dq_dt = 0.5 * np.array([
            -quat[1]*p - quat[2]*q - quat[3]*r,
            quat[0]*p + quat[2]*r - quat[3]*q,
            quat[0]*q - quat[1]*r + quat[3]*p,
            quat[0]*r + quat[1]*q - quat[2]*p
        ])

        return accel_linear, accel_angular, dq_dt