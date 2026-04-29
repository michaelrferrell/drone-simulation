# Attention default units are SI
import numpy as np
from utils import quat_multiply, quat_conjugate, split_quat
from conversions_constants import *

class FlightComputer:
    def __init__(self, attitude_kp, attitude_kd, pos_kp, pos_kd, r_start, v_start, r_end, r_return, t_f, t_hover, arm_length, torque_coeff, mass, payload_threshold):
        # Gains
        self.attitude_kp = attitude_kp
        self.attitude_kd = attitude_kd
        self.pos_kp = pos_kp
        self.pos_kd = pos_kd
        
        # Desired states
        self.r_start = r_start.copy()
        self.v_start = v_start.copy()
        self.r_end = r_end.copy()
        self.r_return = r_return.copy() # Desired position for drone to return to
        
        # Trajectory info
        self.t_f = t_f # Time to reach trajectory endpoint
        self.t_hover = t_hover # Time that drone hovers while delivering payload
        
        # Vehicle information
        self.arm_length = arm_length
        self.torque_coeff = torque_coeff
        self.mass = mass
        self.payload_threshold = payload_threshold
        self.deployed_payload = False

    # compute_motor_commands function
    # Takes sensor readings and user input, outputs raw commands for motors
    def compute_motor_commands(self, sensor_readings, target_acceleration, max_tilt_angle, min_throttle):     
        # Unpack sensor data (process variables)
        quat_fc = sensor_readings['quaternion']   # [w, x, y, z] body->inertial
        omega_fc = sensor_readings['omega']       # [p, q, r] body rate

        if np.linalg.norm(np.cross(np.array([0, 0, 1]), target_acceleration)) != 0:
            quatAxisAngleRotVec = np.cross(np.array([0, 0, 1]), target_acceleration)/np.linalg.norm(np.cross(np.array([0, 0, 1]), target_acceleration))
        else:
            quatAxisAngleRotVec = np.array([0, 0, 1])
        quatAxisAngleRotAngle = np.arccos(np.dot(np.array([0, 0, 1]), target_acceleration)/np.linalg.norm(target_acceleration))
        quatAxisAngleRotAngle = min(quatAxisAngleRotAngle, max_tilt_angle)
        target_quaternion = np.array([np.cos(quatAxisAngleRotAngle/2), quatAxisAngleRotVec[0]*np.sin(quatAxisAngleRotAngle/2), quatAxisAngleRotVec[1]*np.sin(quatAxisAngleRotAngle/2), quatAxisAngleRotVec[2]*np.sin(quatAxisAngleRotAngle/2)])
        error_quat = quat_multiply(quat_conjugate(target_quaternion), quat_fc)
        error_quat_scalar, error_quat_vector = split_quat(error_quat)

        tau_des = -error_quat_scalar*self.attitude_kp@error_quat_vector - self.attitude_kd@omega_fc
        thrust_des = self.mass*np.linalg.norm(target_acceleration)

        mapping_matrix = np.array([[1, 1, 1, 1],
                                   [0, 0, self.arm_length, -self.arm_length],
                                   [-self.arm_length, self.arm_length, 0, 0],
                                   [-self.torque_coeff, -self.torque_coeff, self.torque_coeff, self.torque_coeff]])

        
        m1_cmd, m2_cmd, m3_cmd, m4_cmd = np.linalg.inv(mapping_matrix)@np.array([thrust_des, tau_des[0], tau_des[1], tau_des[2]]).T

        commands = [
            max(0.0, m1_cmd),
            max(0.0, m2_cmd),
            max(0.0, m3_cmd),
            max(0.0, m4_cmd)
        ]
        
        return commands

    # compute_target_acceleration function
    # Returns target acceleration for desired position and velocity
    def compute_target_acceleration(self, sensor_readings, r_des, v_des, a_des):
        # Unpack sensor data (process variables)
        r_fc = sensor_readings['position']      # [x, y, z] inertial
        v_fc = sensor_readings['velocity']      # [vx, vy, vz] inertial
        r_error = r_des - r_fc
        v_error = v_des - v_fc

        target_acceleration = a_des -self.pos_kp@r_error.T -self.pos_kd@v_error.T + np.array([0, 0, STANDARD_GRAVITY]).T
        if target_acceleration[2] < 0:
            target_acceleration[2] = 0
        if np.linalg.norm(target_acceleration) > 2*STANDARD_GRAVITY:
            target_acceleration = target_acceleration/np.linalg.norm(target_acceleration)*2*STANDARD_GRAVITY
        # target_acceleration = np.array([2.0, 0.0, 9.807])
        return target_acceleration
    
    # compute_desired_trajectory function
    # Returns required trajectory to go from starting position to ending position
    def compute_desired_trajectory(self, current_time, t_f, r_start, v_start, r_des):
        M = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 2, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 6, 0, 0, 0, 0],
                      [1, t_f, t_f**2, t_f**3, t_f**4, t_f**5, t_f**6, t_f**7], 
                      [0, 1, 2*t_f, 3*t_f**2, 4*t_f**3, 5*t_f**4, 6*t_f**5, 7*t_f**6],
                      [0, 0, 2, 6*t_f, 12*t_f**2, 20*t_f**3, 30*t_f**4, 42*t_f**5], 
                      [0, 0, 0, 6, 24*t_f, 60*t_f**2, 120*t_f**3, 210*t_f**4]])

        start_state_vec_x = np.array([r_start[0], v_start[0], 0, 0])
        start_state_vec_y = np.array([r_start[1], v_start[1], 0, 0])
        start_state_vec_z = np.array([r_start[2], v_start[2], 0, 0])

        M_inv = np.linalg.inv(M)
        coeffs_x = M_inv@np.array([start_state_vec_x[0], start_state_vec_x[1], start_state_vec_x[2], start_state_vec_x[3], r_des[0], 0.0, 0.0, 0.0]).T
        coeffs_y = M_inv@np.array([start_state_vec_y[0], start_state_vec_y[1], start_state_vec_y[2], start_state_vec_y[3], r_des[1], 0.0, 0.0, 0.0]).T
        coeffs_z = M_inv@np.array([start_state_vec_z[0], start_state_vec_z[1], start_state_vec_z[2], start_state_vec_z[3], r_des[2], 0.0, 0.0, 0.0]).T

        coeffs_x = np.flip(coeffs_x)
        coeffs_y = np.flip(coeffs_y)
        coeffs_z = np.flip(coeffs_z)

        r_des_x = np.polyval(coeffs_x, current_time)
        r_des_y = np.polyval(coeffs_y, current_time)
        r_des_z = np.polyval(coeffs_z, current_time)

        v_des_x = np.polyval(np.polyder(coeffs_x), current_time)
        v_des_y = np.polyval(np.polyder(coeffs_y), current_time)
        v_des_z = np.polyval(np.polyder(coeffs_z), current_time)

        a_des_x = np.polyval(np.polyder(np.polyder(coeffs_x)), current_time)
        a_des_y = np.polyval(np.polyder(np.polyder(coeffs_y)), current_time)
        a_des_z = np.polyval(np.polyder(np.polyder(coeffs_z)), current_time)

        # return np.array([1.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        return np.array([r_des_x, r_des_y, r_des_z]), np.array([v_des_x, v_des_y, v_des_z]), np.array([a_des_x, a_des_y, a_des_z])

    # process_payload_deployment function
    # Checks if target target payload deployment location is reached
    def process_payload_deployment(self, sensor_readings, r_des, threshold):
        r_fc = sensor_readings['position'] # [x, y, z] inertial
        r_error = r_des - r_fc
        distance = np.linalg.norm(r_error)
        
        if distance < threshold and self.deployed_payload == False:
            self.deployed_payload = True
            return "DEPLOY"
        else:
            return "SAFE"