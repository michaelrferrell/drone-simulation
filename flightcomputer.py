# Attention default units are SI
import numpy as np

class FlightComputer:
    def __init__(self):
        # Store gains and internal state for tracking of inegral error and last error here
        pass

    # compute_motor_commands function
    # Takes sensor readings and user input, outputs raw commands for motors
    def compute_motor_commands(self, sensor_readings, target_setpoint, dt):     
        # Unpack sensor data (process variables)
        pos = sensor_readings['position']      # [x, y, z] inertial
        vel = sensor_readings['velocity']      # [vx, vy, vz] inertial
        quat = sensor_readings['quaternion']   # [w, x, y, z] body->inertial
        omega = sensor_readings['omega']       # [p, q, r] body rate

        # Test
        total_thrust_cmd = 10 # (N)
        m1_cmd = total_thrust_cmd / 4.0
        m2_cmd = total_thrust_cmd / 4.0
        m3_cmd = total_thrust_cmd / 4.0
        m4_cmd = total_thrust_cmd / 4.0

        # Formatting
        commands = [
            max(0.0, m1_cmd),
            max(0.0, m2_cmd),
            max(0.0, m3_cmd),
            max(0.0, m4_cmd)
        ]
        
        return commands