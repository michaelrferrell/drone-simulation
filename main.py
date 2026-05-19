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
R_CG = [-0.000032, 0.000011, 0.02277] # Relative to body frame origin (IMU)
R_CP_REF = [0.0, 0.0, 0.0] # Relative to body frame origin (IMU)
ARM_LENGTH = 0.13139 # Relative to CG

# Payload properties
PAYLOAD_MASS = 0.071
ANCHOR_POINT = [0.016, 0.0125, -0.0251] # Relative to body frame origin (IMU)
LOWERING_SPEED = 0.5
MAX_STRING_LENGTH = 1.0

# Inertia tensor
I_XX = 0.003053875
I_YX = 0.000000028
I_ZX = 0.000000833
I_XY = 0.000000028
I_YY = 0.002950495
I_ZY = -0.00001634
I_XZ = 0.000000833
I_YZ = -0.00001634
I_ZZ = 0.003611372
INERTIA = [[I_XX, I_YX, I_ZX], [I_XY, I_YY, I_ZY], [I_XZ, I_YZ, I_ZZ]]

# Motor / propeller characteristics
MAX_THRUST_PER_MOTOR = 8.0442
TORQUE_COEFF         = 0.013771504
MOTOR_LAG            = 0.162462769

# Utility blocks
env = Environment()
dyn = Dynamics()
rk4 = RK4()

# ----------------------------------------------------------------------
# SETUP SYSTEMS
# ----------------------------------------------------------------------
# Motor 1: (+x, +y) - spins CCW
m1 = Motor([0.088701, 0.088957, 0], [0,0,1], -TORQUE_COEFF, MAX_THRUST_PER_MOTOR, MOTOR_LAG)

# Motor 2: (+x, -y) - spins CW
m2 = Motor([0.088936, -0.088658, 0], [0,0,1], TORQUE_COEFF, MAX_THRUST_PER_MOTOR, MOTOR_LAG)

# Motor 3: (-x, -y) - spins CCW
m3 = Motor([-0.088679, -0.088893, 0], [0,0,1], -TORQUE_COEFF, MAX_THRUST_PER_MOTOR, MOTOR_LAG)

# Motor 4: (-x, +y) - spins CW
m4 = Motor([ -0.088914, 0.088722, 0], [0,0,1], TORQUE_COEFF, MAX_THRUST_PER_MOTOR, MOTOR_LAG)

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
v_end = np.array([0.0, 0.0, 0.0]) # Payload delivery target velocity
r_return = np.array([9.0, 9.0, 1.0]) # Return coordinates for drone
r_threshold = 0.3
v_threshold = 0.1
t_f = 5 # Desired time to payload delivery position
t_hover = 2 # Time maintaining payload delivery position

attitude_kp = np.array([[2, 0, 0],
                        [0, 2, 0],
                        [0, 0, 0.01]])

attitude_kd = np.array([[0.4, 0, 0],
                        [0, 0.4, 0],
                        [0, 0, 0.01]])

pos_kp = np.array([[-2, 0, 0],
                   [0, -2, 0],
                   [0, 0, -20]])

pos_kd = np.array([[-5, 0, 0],
                   [0, -5, 0],
                   [0, 0, -10]])

fc = FlightComputer(attitude_kp, attitude_kd, pos_kp, pos_kd, r_start, v_start, r_end, v_end, r_return, t_f, t_hover, ARM_LENGTH, TORQUE_COEFF, VEHICLE_MASS, r_threshold, v_threshold)

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