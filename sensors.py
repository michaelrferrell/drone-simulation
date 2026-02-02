# Attention default units are SI
import numpy as np

class Sensors:
    def __init__(self):
        # Initialize sensor parameters
        # Add noise standard deviations, biases, and sample rates here
        pass
        
    # measure function
    # Takes the true physical state and simulates sensor readings
    def measure(self, true_state):
        # GPS / motion capture (position)
        measured_pos = true_state.position.copy()
        
        # GPS / velocity estimate
        measured_vel = true_state.velocity.copy()
        
        # Attitude reference system (orientation)
        measured_quat = true_state.quaternion.copy()
        
        # Gyroscope (angular velocity)
        measured_omega = true_state.omega.copy()
        
        # Pack into a dictionary for the flight computer
        readings = {
            'position': measured_pos,
            'velocity': measured_vel,
            'quaternion': measured_quat,
            'omega': measured_omega
        }
        
        return readings