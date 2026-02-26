# Attention default units are SI
import numpy as np
import pandas as pd

class Simulation:
    def __init__(self, duration, dt, vehicle, propulsion, flight_computer, sensors, state, dynamics, solver, environment, bounds):
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
        
        for step in range(total_steps):
    
            if step == 0: #UGLY, FIX
                motor_commands = np.array([0.0, 0.0, 0.0, 0.0])
            # Act
            self.prop.update(motor_commands, self.dt)
            
            # Evolve
            derivatives = self.solver.step(self.state, self.vehicle, self.prop, self.dynamics, self.env, self.dt)

            # Sense
            omega = self.state.omega
            accel = derivatives[0]
            sensor_readings = self.sensors.measure(self.state.copy(), omega, accel, self.dt)
            # Think
            # replace with trajgen outputs
            r_des = np.array([3.0, -2.0, 3.0])
            v_des = np.array([0.0, 0.0, 0.0])
            a_des = np.array([0.0, 0.0, 0.0])
            target_quaternion = self.fc.compute_target_acceleration(sensor_readings, r_des, v_des, a_des)  # replace with r_des, v_des, a_des from trajectory
            motor_commands = self.fc.compute_motor_commands(sensor_readings, target_quaternion, 90*np.pi/180, 0.1)
            


            # Safety check
            if self.check_safety_violation():
                break
            
            # Log
            self.log_step(motor_commands)
            
            # Advance time
            self.time += self.dt
            self.step_count += 1
            
        print("Simulation Complete.")
        
        # Return results
        return pd.DataFrame(self.history)

    # log_step function
    # Internal helper to pack current state into a dictionary
    def log_step(self, motor_commands):
        # Extract individual components for cleaner columns
        pos = self.state.position
        vel = self.state.velocity
        quat = self.state.quaternion
        omega = self.state.omega
        
        actual_thrusts = [m.current_thrust for m in self.prop.prop_devices]
        
        log_entry = {
            'time': self.time,
            
            # Position
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            
            # Velocity
            'vx': vel[0], 'vy': vel[1], 'vz': vel[2],
            
            # Orientation (Quaternion)
            'qw': quat[0], 'qx': quat[1], 'qy': quat[2], 'qz': quat[3],
            
            # Angular Velocity
            'p': omega[0], 'q': omega[1], 'r': omega[2],
            
            # Commands (Inputs)
            'cmd_m1': motor_commands[0],
            'cmd_m2': motor_commands[1],
            'cmd_m3': motor_commands[2],
            'cmd_m4': motor_commands[3],
            
            # Actual Thrust (Actuators)
            'thrust_m1': actual_thrusts[0],
            'thrust_m2': actual_thrusts[1],
            'thrust_m3': actual_thrusts[2],
            'thrust_m4': actual_thrusts[3]
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