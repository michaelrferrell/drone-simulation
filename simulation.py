# Attention default units are SI
import numpy as np
import pandas as pd

class Simulation:
    def __init__(self, duration, dt, vehicle, propulsion, flight_computer, sensors, state, dynamics, solver, environment, payload, bounds):
        # Simulation setup 
        self.duration = duration
        self.dt = dt
        
        # System blocks
        self.vehicle = vehicle
        self.prop = propulsion
        self.fc = flight_computer
        self.sensors = sensors
        self.state = state
        self.dynamics = dynamics
        self.solver = solver
        self.env = environment
        self.payload = payload
        
        # Internal counters
        self.time = 0.0
        self.step_count = 0
        
        # Safety limits
        self.bounds = bounds
        self.ground_z = bounds.get('min_z', 0.0)
        self.crash_velocity_threshold = -2.0  # (m/s)
        
        # Data logging container
        self.history = []

    # run function
    # Executes the main simulation loop and returns the data log
    def run(self):
        # Calculate total steps
        total_steps = int(self.duration / self.dt)
        
        print(f"Starting Simulation: {self.duration}s ({total_steps} steps)")
        
        # Pre-Loop initialization (t=0)
        # Sense
        omega = self.state.omega
        accel = np.array([0.0, 0.0, 0.0]) # Initial acceleration at rest
        sensor_readings = self.sensors.measure(self.state.copy(), omega, accel, self.dt)
        
        target_acceleration = self.fc.compute_target_acceleration(sensor_readings, self.fc.r_start, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))  # replace with r_des, v_des, a_des from trajectory
        motor_commands = self.fc.compute_motor_commands(sensor_readings, target_acceleration, 90*np.pi/180, 0.1)
        payload_command = self.fc.process_payload_deployment(sensor_readings, self.fc.r_end, self.fc.payload_threshold, self.fc.payload_mass)
        
        # Log initial state at t=0
        self.log_step(motor_commands)
        
        # Main simulation loop
        for step in range(total_steps):
            # Act
            self.prop.update(motor_commands, self.dt)
            if payload_command == "DEPLOY":
                self.vehicle.update_mass(-self.payload.mass)
            
            # Evolve
            derivatives = self.solver.step(self.state, self.vehicle, self.prop, self.dynamics, self.env, self.dt)
            
            # Advance time (t + dt)
            self.time += self.dt
            self.step_count += 1

            # Sense
            omega = self.state.omega
            accel = derivatives[0]
            sensor_readings = self.sensors.measure(self.state.copy(), omega, accel, self.dt)
            
            # Think
            t_f = 3.0
            # print(self.fc.r_start - self.state.position)
            if self.time < t_f:
                r_des, v_des, a_des = self.fc.compute_desired_trajectory(self.time, t_f)
            elif self.time > t_f + 5.0:
                r_des = np.array([0, 0, 2.0])
                v_des = np.array([0, 0, 0])
                a_des = np.array([0, 0, 0])
            # r_des = np.array([3*np.sin(self.time), 3*np.cos(self.time), 5.0])
            # v_des = np.array([3*np.cos(self.time), -3*np.sin(self.time), 0.0])
            # a_des = np.array([0.0, 0.0, 0.0])
    
            target_acceleration = self.fc.compute_target_acceleration(sensor_readings, r_des, v_des, a_des)  # replace with r_des, v_des, a_des from trajectory
            motor_commands = self.fc.compute_motor_commands(sensor_readings, target_acceleration, 90*np.pi/180, 0.1)


            payload_command = self.fc.process_payload_deployment(sensor_readings, self.fc.r_end, self.fc.payload_threshold, self.fc.payload_mass)
            
            # Safety check
            if self.check_safety_violation():
                break
            
            # Log
            self.log_step(motor_commands, r_des=r_des)
            
        print("Simulation Complete.")
        
        # Return results
        return pd.DataFrame(self.history)

    # log_step function
    # Internal helper to pack current state into a dictionary
    def log_step(self, motor_commands, r_des=None):
        # Extract individual components for cleaner columns
        pos = self.state.position
        vel = self.state.velocity
        quat = self.state.quaternion
        omega = self.state.omega
        mass = self.vehicle.mass

        if r_des is not None:
            traj = r_des
        else:
            traj = np.array([0, 0, 0])
        
        actual_thrusts = [m.current_thrust for m in self.prop.prop_devices]
        
        log_entry = {
            'time': self.time,
            
            # Position
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            
            # Velocity
            'vx': vel[0], 'vy': vel[1], 'vz': vel[2],
            
            # Orientation (quaternion)
            'qw': quat[0], 'qx': quat[1], 'qy': quat[2], 'qz': quat[3],
            
            # Angular velocity
            'p': omega[0], 'q': omega[1], 'r': omega[2],
            
            # Commands (inputs)
            'cmd_m1': motor_commands[0],
            'cmd_m2': motor_commands[1],
            'cmd_m3': motor_commands[2],
            'cmd_m4': motor_commands[3],
            
            # Actual thrust (actuators)
            'thrust_m1': actual_thrusts[0],
            'thrust_m2': actual_thrusts[1],
            'thrust_m3': actual_thrusts[2],
            'thrust_m4': actual_thrusts[3],
            
            # Vehicle mass
            'mass': mass,

            # Trajectory
            'x_des': traj[0], 'y_des': traj[1], 'z_des': traj[2],
                
        }
        
        self.history.append(log_entry)
        
    # check_safety_violation function
    # Checks if that state has violated physical boundaries
    def check_safety_violation(self):
        # Geofence check
        dist = np.linalg.norm(self.state.position)
        if dist > self.bounds['max_dist']:
            print(f"!!! FAIL: Drone left simulation area (Dist={dist:.1f}m) !!!")
            return True

        # Ground interaction check
        if self.state.position[2] < self.ground_z:
            current_vz = self.state.velocity[2]
            
            # Hard impact case
            if current_vz < self.crash_velocity_threshold:
                print(f"!!! CRASH: Hit ground at {current_vz:.2f} m/s !!!")
                self.log_step([0,0,0,0]) # Log the impact
                return True # stop simulation
            
            # Soft landing case
            else:
                # Clamp position
                self.state.position[2] = self.ground_z
                
                # Zero out downward velocity
                if self.state.velocity[2] < 0:
                     self.state.velocity[2] = 0.0
                     
                     # Friction slows lateral movement
                     self.state.velocity[0] *= 0.95 
                     self.state.velocity[1] *= 0.95
                     
                     # Kill rotation if on ground (prevent tipping)
                     self.state.omega *= 0.9
                
                # The sim continues (drone is just sitting)
                return False

        return False