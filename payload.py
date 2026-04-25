# Attention default units are SI
import numpy as np
from conversions_constants import *

class Payload:
    def __init__(self, mass, anchor_point, dl_dt, max_length):
        # Physical properties
        self.mass = mass # [kg]
        self.anchor_point = np.array(anchor_point) # [x, y, z] in drone body frame
        self.dl_dt = dl_dt # Constant lowering speed [m/s]
        self.max_length = max_length # [m]
        self.state = np.zeros(6) # [l, theta, phi, l_dot, theta_dot, phi_dot]
        
        # Deployment status
        self.status = "STOWED" # States: STOWED, LOWERING, FREEFALL, DROPPED
        
        # Freefall variables
        self.freefall_pos = None
        self.freefall_vel = None
        
    # trigger_deployment function
    # Begins payload lowering process by setting dl_dt to non-zero    
    def trigger_deployment(self):
        if self.status == "STOWED":
            self.status = "LOWERING"
            self.state[3] = self.dl_dt # Set l_dot
            
    # compute_derivatives function
    # Computes derivatives of payload state variables
    def compute_derivatives(self, drone_accel_inertial, environment):
        # If STOWED or DROPPED variables are zero
        if self.status in ["STOWED", "DROPPED"]:
            return np.zeros(6)
            
        # If LOWERING
        l, theta, phi, l_dot, theta_dot, phi_dot = self.state
        safe_l = max(l, 1e-4)
        
        # Calculate net effective acceleration acting on egg
        gravity_vector = - np.array([0, 0, environment.g])
        net_accel = gravity_vector - drone_accel_inertial
        ax, ay, az = net_accel[0], net_accel[1], net_accel[2]
        
        # Polar angle ODE
        theta_ddot = (np.sin(theta) * np.cos(theta) * phi_dot**2) - \
                     (2.0 * l_dot / safe_l) * theta_dot + \
                     (1.0 / safe_l) * (ax * np.cos(theta) * np.cos(phi) + 
                                       ay * np.cos(theta) * np.sin(phi) + 
                                       az * np.sin(theta))
        theta_ddot -= 0.5 * theta_dot # 0.5 is arbitrary damping coefficient
                     
        # Azimuthal angle ODE
        safe_sin_theta = np.sin(theta)
        
        # Neutralize coordinate singularity at the pole
        if abs(safe_sin_theta) < 0.05:
            cot_theta = 0.0 
            phi_accel_term = 0.0
            phi_dot = 0.0
        else:
            cot_theta = np.cos(theta) / safe_sin_theta
            phi_accel_term = (1.0 / (safe_l * safe_sin_theta)) * (-ax * np.sin(phi) + ay * np.cos(phi))
            
        phi_ddot = - (2.0 * l_dot / safe_l) * phi_dot - \
                   (2.0 * theta_dot * phi_dot * cot_theta) + \
                   phi_accel_term 
        phi_ddot -= 0.5 * phi_dot # 0.5 is arbitrary damping coefficient
        
        # Hard clamp to protect RK4 integrator
        theta_ddot = np.clip(theta_ddot, -100.0, 100.0)
        phi_ddot = np.clip(phi_ddot, -100.0, 100.0)
                   
        # Constant lowering speed
        l_ddot = 0.0
        
        return np.array([l_dot, theta_dot, phi_dot, l_ddot, theta_ddot, phi_ddot])
    
    # compute_wrench function
    # Calculates the force and torque applied to the drone by the payload in the drone's body frame
    def compute_wrench(self, drone_accel_body, drone_state, environment):
        # If in FREEFALL or DROPPED imparts no force or torque on drone
        if self.status in ["FREEFALL", "DROPPED"]:
            return np.zeros(3), np.zeros(3)
        
        # If STOWED acts as a fixed mass on the drone    
        elif self.status == "STOWED":
            gravity_inertial = -np.array([0.0, 0.0, environment.g])
            R_inertial_to_body = drone_state.get_rotation_matrix().T
            gravity_body = R_inertial_to_body.dot(gravity_inertial)
            
            force_body = self.mass * (gravity_body - drone_accel_body)
            torque_body = np.cross(self.anchor_point, force_body)
            
            return force_body, torque_body
        
        # If LOWERING tension of holding egg on string imparts force and moment on egg
        elif self.status == "LOWERING":
            l, theta, phi, l_dot, theta_dot, phi_dot = self.state
            
            # Calculate magnitude of the tension in the string.
            tension_mag = self.mass * (environment.g * np.cos(theta) + 
                                       l * theta_dot**2 + 
                                       l * (np.sin(theta)**2) * phi_dot**2)
            
            # Convert tension into 3D force vector in inertial frame
            direction_inertial = np.array([ # The string pulls ON the drone, towards the egg
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                -np.cos(theta) ])
            force_inertial = tension_mag * direction_inertial 
            
            # Convert the force into drone's body frame
            R_body_to_inertial = drone_state.get_rotation_matrix()
            R_inertial_to_body = R_body_to_inertial.T
            force_body = R_inertial_to_body.dot(force_inertial)
            
            # Calculate torque applied to drone frame
            torque_body = np.cross(self.anchor_point, force_body)
            
            return force_body, torque_body