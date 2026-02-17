# Attention default units are SI
import numpy as np

class Sensors:
    def __init__(self, state):
        self.omega = np.array([0.0, 0.0, 0.0])
        self.accel = np.array([0.0, 0.0, 0.0])

        self.vel = state.velocity
        self.pos = state.position
        self.quat = state.quaternion
        # Initialize sensor parameters
        # Add noise standard deviations, biases, and sample rates here
        pass
        
    # measure function
    # Takes the true physical state and simulates sensor readings
    def measure(self, omega, accel, dt):
        # GPS / motion capture (position)
        self.accel = accel + np.random.uniform(-1, 1)
        self.omega = omega + np.random.uniform(-5*np.pi/180, 5*np.pi/180)

        self.vel += self.accel*dt
        self.pos += self.vel*dt
        
        # Attitude reference system (orientation)
        quat = self.quat
        p, q, r = self.omega
        dq_dt = 0.5 * np.array([
            -quat[1]*p - quat[2]*q - quat[3]*r,
            quat[0]*p + quat[2]*r - quat[3]*q,
            quat[0]*q - quat[1]*r + quat[3]*p,
            quat[0]*r + quat[1]*q - quat[2]*p
        ])
        self.quat += dq_dt*dt
        self.quat = self.quat/np.linalg.norm(self.quat)


        # Gyroscope (angular velocity)
        
        # Pack into a dictionary for the flight computer
        readings = {
            'position': self.pos,
            'velocity': self.vel,
            'quaternion': self.quat,
            'omega': self.omega
        }
        
        return readings