# Attention default units are SI
# ----------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------
from utils import *
from simulation import Simulation
from vehicle import Vehicle
from propulsion import Propulsion, Motor
from state import State
from flightcomputer import FlightComputer
from sensors import Sensors
from dynamics import Dynamics
from solver import RK4
from environment import Environment

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# Exports
plot_results = True
export_results = False
animate = True

# Time
DURATION = 10.0
DT       = 0.01

# Physical properties
MASS = 0.9 # Total mass of vehicle (including motors)
R_CG = [0.0, 0.0, 0.0] # Relative to body frame origin
R_CP_REF = [0.0, 0.0, 0.0] # Relative to body frame origin
ARM_LENGTH = 0.1

# Inertia tensor
I_XX = 0.004
I_YY = 0.004
I_ZZ = 0.004
INERTIA = [[I_XX, 0, 0], [0, I_YY, 0], [0, 0, I_ZZ]]

# Motor / propeller characteristics
MAX_THRUST_PER_MOTOR = 5.0
TORQUE_COEFF         = 0.001
MOTOR_LAG            = 0.05

# Utility blocks
env = Environment()
dyn = Dynamics()
rk4 = RK4()
sensors = Sensors()

# ----------------------------------------------------------------------
# SETUP SYSTEMS
# ----------------------------------------------------------------------
# Motor 1: (+x, 0) - spins CCW
m1 = Motor([ARM_LENGTH, 0, 0], [0,0,1], -TORQUE_COEFF, MAX_THRUST_PER_MOTOR, MOTOR_LAG)

# Motor 2: (-x, 0) - spins CCW
m2 = Motor([-ARM_LENGTH, 0, 0], [0,0,1],  -TORQUE_COEFF, MAX_THRUST_PER_MOTOR, MOTOR_LAG)

# Motor 3: (0, +y) - spins CW
m3 = Motor([0, ARM_LENGTH, 0], [0,0,1], TORQUE_COEFF, MAX_THRUST_PER_MOTOR, MOTOR_LAG)

# Motor 4: (0, -y) - spins CW
m4 = Motor([ 0, -ARM_LENGTH, 0], [0,0,1],  TORQUE_COEFF, MAX_THRUST_PER_MOTOR, MOTOR_LAG)

prop_system = Propulsion([m1, m2, m3, m4])

# Vehicle body
vehicle = Vehicle(MASS, INERTIA, R_CG, R_CP_REF)

# Initial state
initial_state = State(
    position   = [0.0, 0.0, 1.0],
    velocity   = [0.0, 0.0, 0.0],
    quaternion = [1.0, 0.0, 0.0, 0.0],
    omega      = [0.0, 0.0, 0.0]
)

# Flight computer
attitude_kp = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0.01]])
attitude_kd = np.array([[0.1, 0, 0],
                        [0, 0.1, 0],
                        [0, 0, 0.01]])
pos_kp = np.array([[-3, 0, 0],
                   [0, -3, 0],
                   [0, 0, -20]])
pos_kd = np.array([[-4, 0, 0],
                   [0, -4, 0],
                   [0, 0, -10]])

fc = FlightComputer(attitude_kp, attitude_kd, pos_kp, pos_kd, ARM_LENGTH, TORQUE_COEFF, MASS)

# ----------------------------------------------------------------------
# RUN SIMULATION
# ----------------------------------------------------------------------
safety_bounds = {
    'min_z': 0.0,      # Ground level
    'max_dist': 20.0   # Geofence radius
}

sim = Simulation(
    duration=DURATION,
    dt=DT,
    vehicle=vehicle,
    propulsion=prop_system,
    flight_computer=fc,
    sensors=sensors,
    state=initial_state,
    dynamics=dyn,
    solver=rk4,
    environment=env,
    bounds=safety_bounds
)

print("Running Simulation...")
df = sim.run()

# ----------------------------------------------------------------------
# EXPORT DATA
# ---------------------------------------------------------------------- 
if export_results:
    sim_metadata = {
        "duration": DURATION,
        "dt": DT,
        "mass": MASS,
        "inertia": INERTIA,
        "motor_lag": MOTOR_LAG,
        "max_thrust": MAX_THRUST_PER_MOTOR,
        "initial_pos": initial_state.position
    }

    export_simulation_data(df, sim_metadata)
    
# ----------------------------------------------------------------------
# PLOT DATA
# ----------------------------------------------------------------------
if plot_results:
    plot_simulation_results(df, max_thrust_limit=MAX_THRUST_PER_MOTOR)
    
# ----------------------------------------------------------------------
# ANIMATE
# ----------------------------------------------------------------------
if animate:
    animate_simulation_3d(df)