# Attention default units are SI
import numpy as np
from abc import ABC, abstractmethod

class Propulsion:
    def __init__(self, prop_devices):
        self.prop_devices = prop_devices

    # update function
    # Updates the internal state (lag) of all propulsion devices
    def update(self, commands, dt):
        # Ensure there are enough commands for devices
        if len(commands) != len(self.prop_devices):
            raise ValueError(f"Expected {len(self.prop_devices)} commands, got {len(commands)}")

        for i, device in enumerate(self.prop_devices):
            device.update(commands[i], dt)

    # compute_wrench function
    # Aggregates the total force and torque from all devices
    def compute_wrench(self, r_cg):
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        # Each device calculates its own contribution based on where the CG is
        for device in self.prop_devices:
            f, t = device.get_wrench(r_cg)
            total_force += f
            total_torque += t
            
        return total_force, total_torque

class PropulsionDevice(ABC):
    def __init__(self):
        self.current_thrust = 0.0
    
    @abstractmethod
    # update function
    # Updates motor state based on the command and time step
    def update(self, command, dt):
        """Must update internal state based on command"""
        pass
        
    # get_wrench function
    # Calculates the force and torque this motor applies to the vehicle
    @abstractmethod
    def get_wrench(self, r_cg):
        """Must return (force_vector, torque_vector)"""
        pass

class Motor(PropulsionDevice):
    def __init__(self, position, direction, torque_coeff, max_thrust, time_constant_tau):
        # Call parent init
        super().__init__()
        
        # Position [x, y, z] relative to body frame origin
        self.position = np.array(position)
        
        # Direction [x, y, z] unit vector pointing in direction of thrust
        self.direction = np.array(direction) / np.linalg.norm(direction)
        
        # Torque coefficient scalar linking Thrust (N) to Torque (Nm)
        self.torque_coeff = torque_coeff # Positive/Negative determines CW/CCW torque reaction
        
        # Max thrust of motor+propeller (N)
        self.max_thrust = max_thrust
        
        # Time constant (s)
        self.tau = time_constant_tau

    # update function
    # Updates motor state based on the command and time step
    def update(self, commanded_thrust, dt):
        # Clamp command to physical limits (0 to Max)
        cmd_clamped = np.clip(commanded_thrust, 0.0, self.max_thrust)
        
        # Calculate derivative
        # T_dot = (T_cmd - T_current) / tau
        thrust_rate = (cmd_clamped - self.current_thrust) / self.tau
        
        # Integrate to get new actual thrust
        self.current_thrust += thrust_rate * dt
        
        return self.current_thrust
    
    # get_wrench function
    # Calculates the force and torque this motor applies to the vehicle
    def get_wrench(self, r_cg):
        T = self.current_thrust
        
        # Force vector
        force = T * self.direction
        
        # Torque vector
        lever_arm = self.position - r_cg
        torque_thrust = np.cross(lever_arm, force)
        
        # Moment from propeller drag acting along the thrust axis
        torque_drag = (T * self.torque_coeff) * self.direction
        
        return force, torque_thrust + torque_drag