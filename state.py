# Attention default units are SI
import numpy as np

class State:
    def __init__(self, position, velocity, quaternion, omega):
        # Position [x, y, z] (inertial frame)
        self.position = np.array(position)
        
        # Velocity [vx, vy, vz] (inertial frame)
        self.velocity = np.array(velocity)
        
        # Quaternion [w, x, y, z] (orientation from body -> inertial)
        self.quaternion = np.array(quaternion)
        
        # Angular velocity [p, q, r] (body frame)
        self.omega = np.array(omega)

    # get_rotation_matrix function
    # Converts current quaternion state to 3x3 rotation matrix (body -> inertial)
    def get_rotation_matrix(self):
        w, x, y, z = self.quaternion

        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])
        
        return R

    # integrate function
    # Updates the state using Euler integration and ensures quaternion validity
    def integrate(self, derivatives, dt):
        """
        Updates:
        1. position += velocity * dt
        2. velocity += linear_accel * dt
        3. quaternion += q_dot * dt (Includes Normalization)
        4. omega += angular_accel * dt
        """
        accel_linear, accel_angular, dq_dt = derivatives

        # Update velocity
        self.velocity += accel_linear * dt

        # Update position
        self.position += self.velocity * dt



        # Update orientation (quaternion)
        self.quaternion += dq_dt * dt
        
        # Normalization
        norm = np.linalg.norm(self.quaternion)
        if norm > 0:
            self.quaternion /= norm

        # Update Angular Velocity
        self.omega += accel_angular * dt
    
    # copy function
    # Create a new State object with independent copies of arrays
    def copy(self):
        return State(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            omega=self.omega.copy()
        )