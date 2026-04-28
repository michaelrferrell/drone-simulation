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
VEHICLE_MASS = 0.772
R_CG = [float(0.0671/1000), float(-0.113/1000), float(25.092/1000)] # Relative to body frame origin (IMU)
R_CP_REF = [0.0, 0.0, 0.0] # Relative to body frame origin (IMU)
ARM_LENGTH = 0.13139 # Relative to CG

# Payload properties
PAYLOAD_MASS = 0.071
ANCHOR_POINT = [float(16/1000), float(12.5/1000), float(-25.1/1000)] # Relative to body frame origin (IMU)
LOWERING_SPEED = 0.5
MAX_STRING_LENGTH = 1.0

# Inertia tensor
I_XX = 0.002300652
I_YX = 0.000000048
I_ZX = -0.000002475
I_XY = 0.000000048
I_YY = 0.00220791
I_ZY = -0.000004592
I_XZ = -0.000002475
I_YZ = -0.000004592
I_ZZ = 0.00360216
INERTIA = [[I_XX, I_YX, I_ZX], [I_XY, I_YY, I_ZY], [I_XZ, I_YZ, I_ZZ]]

# Motor / propeller characteristics
MAX_THRUST_PER_MOTOR = 8.0442 #5.0 
TORQUE_COEFF         = 0.013771504 #0.001
MOTOR_LAG            = 0.162462769 #0.05

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
vehicle = Vehicle(VEHICLE_MASS, INERTIA, R_CG, R_CP_REF)

# Initial state
initial_state = State(
    position   = [0.0, 0.0, 0.0],
    velocity   = [0.0, 0.0, 0.0],
    quaternion = [1.0, 0.0, 0.0, 0.0],
    omega      = [0.0, 0.0, 0.0]
)

# Flight computer
r_start = np.asarray(initial_state.position)
v_start = np.asarray(initial_state.velocity)
r_end = np.array([10.0, 10.0, 1.0]) # Payload delivery coordinates
r_return = np.array([0.0, 0.0, 5.0]) # Return coordinates for drone
t_f = 5 # Desired time to payload delivery position
t_hover = 0.001 # Time maintaining payload delivery position

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

fc = FlightComputer(attitude_kp, attitude_kd, pos_kp, pos_kd, r_start, v_start, r_end, r_return, t_f, t_hover, ARM_LENGTH, TORQUE_COEFF, VEHICLE_MASS, 0.1)

# Sensors
sensors = Sensors(initial_state.copy())

# Payload
payload = Payload(
    mass=PAYLOAD_MASS,
    anchor_point=np.array(ANCHOR_POINT),
    dl_dt=LOWERING_SPEED,
    max_length=MAX_STRING_LENGTH
)

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
        "mass": VEHICLE_MASS,
        "inertia": INERTIA,
        "motor_lag": MOTOR_LAG,
        "max_thrust": MAX_THRUST_PER_MOTOR,
        "initial_pos": initial_state.position
    }

    export_simulation_data(df, sim_metadata, r'C:\Users\micha\OneDrive\Desktop\outputs\data')
    
# ----------------------------------------------------------------------
# PLOT DATA
# ----------------------------------------------------------------------
if plot_results:
    plot_simulation_results(df, max_thrust_limit=MAX_THRUST_PER_MOTOR)
    
# ----------------------------------------------------------------------
# ANIMATE
# ----------------------------------------------------------------------
if animate:
    if export_results:
        animate_simulation_3d(df, [df['x_des'], df['y_des'], df['z_des']], filename=r'C:\Users\micha\OneDrive\Desktop\outputs\animations\test_animation.gif')
    else:
        animate_simulation_3d(df, [df['x_des'], df['y_des'], df['z_des']])