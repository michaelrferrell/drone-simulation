# Attention default units are SI
import numpy as np

class RK4:
    def __init__(self):
        pass
    
    # step function
    # Performs one RK4 integration step to update the state
    def step(self, state, vehicle, propulsion, dynamics, environment, dt):       
        # Calculate propulsive wrench (force & torque) - assumed constant for the duration of the single time step
        forces_body, torques_body = propulsion.compute_wrench(vehicle.r_cg)
        
        # k1
        k1 = dynamics.compute_derivatives(state, vehicle, forces_body, torques_body, environment)
        
        # k2
        state_k2 = state.copy()
        state_k2.integrate(k1, dt / 2.0)
        k2 = dynamics.compute_derivatives(state_k2, vehicle, forces_body, torques_body, environment)
        
        # k3
        state_k3 = state.copy()
        state_k3.integrate(k2, dt / 2.0)
        k3 = dynamics.compute_derivatives(state_k3, vehicle, forces_body, torques_body, environment)
        
        # k4
        state_k4 = state.copy()
        state_k4.integrate(k3, dt)
        k4 = dynamics.compute_derivatives(state_k4, vehicle, forces_body, torques_body, environment)
        
        # Calculate weighted average derivative (k1 + 2*k2 + 2*k3 + k4) / 6
        final_derivatives = []
        
        # Iterate over the tuple (accel_linear, accel_angular, dq_dt)
        for i in range(3):
            avg = (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0
            final_derivatives.append(avg)
            
        # Apply final integration to the real state
        state.integrate(final_derivatives, dt)
        return final_derivatives