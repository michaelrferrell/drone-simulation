# Attention default units are SI
import numpy as np
from utils import quat_multiply, quat_conjugate, split_quat
from conversions_constants import *

class FlightComputer:
    def __init__(self, attitude_kp, attitude_kd, pos_kp, pos_kd, arm_length, torque_coeff, mass):
        # Store gains and internal state for tracking of inegral error and last error here
        self.attitude_kp = attitude_kp
        self.attitude_kd = attitude_kd
        self.pos_kp = pos_kp
        self.pos_kd = pos_kd
        self.arm_length = arm_length
        self.torque_coeff = torque_coeff
        self.mass = mass

    # compute_motor_commands function
    # Takes sensor readings and user input, outputs raw commands for motors
    def compute_motor_commands(self, sensor_readings, target_acceleration, max_tilt_angle, min_throttle):     
        # Unpack sensor data (process variables)
        quat_fc = sensor_readings['quaternion']   # [w, x, y, z] body->inertial
        omega_fc = sensor_readings['omega']       # [p, q, r] body rate


        quatAxisAngleRotVec = np.cross(np.array([0, 0, 1]), target_acceleration)/np.linalg.norm(np.cross(np.array([0, 0, 1]), target_acceleration))
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

        # Formatting
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

        target_acceleration = 0 -self.pos_kp@r_error.T -self.pos_kd@v_error.T + np.array([0, 0, STANDARD_GRAVITY]).T
        if target_acceleration[2] < 0:
            target_acceleration[2] = 0
        if np.linalg.norm(target_acceleration) > 2*STANDARD_GRAVITY:
            target_acceleration = target_acceleration/np.linalg.norm(target_acceleration)*20
        return target_acceleration