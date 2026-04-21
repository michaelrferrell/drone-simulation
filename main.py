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
from payload import Payload

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# Exports
plot_results = True
animate = True
export_results = False

# Time
DURATION = 15.0
DT       = 0.01

# Physical properties
PAYLOAD_MASS = 0.05
VEHICLE_MASS = 0.9
MASS = VEHICLE_MASS + PAYLOAD_MASS # Total mass of vehicle (including motors)
R_CG = [0.0, 0.0, 0.0] # Relative to body frame origin
R_CP_REF = [0.0, 0.0, 0.0] # Relative to body frame origin
ARM_LENGTH = 0.1

# Inertia tensor
I_XX = 0.004
I_YY = 0.004
I_ZZ = 0.004
INERTIA = [[I_XX, 0, 0], [0, I_YY, 0], [0, 0, I_ZZ]]

# Motor / propeller characteristics
MAX_THRUST_PER_MOTOR = 8.0442
TORQUE_COEFF         = 0.013771504
MOTOR_LAG            = 0.303062563

# Utility blocks
env = Environment()
dyn = Dynamics()
rk4 = RK4()

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
    position   = [1.0, -2.0, 3.0],
    velocity   = [0.0, -0.0, 0.0],
    quaternion = [1.0, 0.0, 0.0, 0.0],
    omega      = [0.0, 0.0, 0.0]
)

# Flight computer
r_start = np.asarray(initial_state.position)
v_start = np.asarray(initial_state.velocity)
r_end = np.array([9.0, 3.0, 1.0])

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

fc = FlightComputer(attitude_kp, attitude_kd, pos_kp, pos_kd, r_start, v_start, r_end, ARM_LENGTH, TORQUE_COEFF, MASS, PAYLOAD_MASS, 0.1)

# Sensors
sensors = Sensors(initial_state.copy())

# Payload
payload = Payload(PAYLOAD_MASS)

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
    payload=payload,
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

    export_simulation_data(df, sim_metadata, 'outputs/data')
    
# ----------------------------------------------------------------------
# PLOT DATA
# ----------------------------------------------------------------------
if plot_results:
    plot_simulation_results(df, max_thrust_limit=MAX_THRUST_PER_MOTOR)
    
# ----------------------------------------------------------------------
# ANIMATE
# ----------------------------------------------------------------------
if animate:
    animate_simulation_3d(df, [df['x_des'], df['y_des'], df['z_des']],filename='outputs/animations/test_animation.gif')