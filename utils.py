import os
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.animation as animation
from functools import partial

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
# _sanitize_params_for_json function
# Helper to convert numpy arrays to lists for JSON dumping
def _sanitize_params_for_json(params):
    clean = {}
    for k, v in params.items():
        if hasattr(v, 'tolist'): # Check if it's a numpy array
            clean[k] = v.tolist()
        else:
            clean[k] = v
    return clean

# quat_to_euler function
# Converts quaternions to standard euler angles
def quat_to_euler(row):
        w, x, y, z = row['qw'], row['qx'], row['qy'], row['qz']
        
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
    
        # Pitch
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
    
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
    
        return pd.Series([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])
    
# quat_multiply function
# Performs quaternion multiplication
def quat_multiply(q1, q2):
    w1, x1, y1, z1, w2, x2, y2, z2 = q1[0], q1[1], q1[2], q1[3], q2[0], q2[1], q2[2], q2[3]
    output_quat = np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*x2])
    return output_quat

# quat_conjugate function
# Returns the conjugate of a quaternion
def quat_conjugate(q1):
    output_quat = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    return output_quat

# split_quat function
# Splits quaternion into its scalar (real) part and vector (imaginary) part
def split_quat(q1):
    return q1[0], np.array([q1[1], q1[2], q1[3]])

# get_orientation_vectors function
# Calculates the Forward (X) and Right (Y) direction vectors from the current quaternion
def get_orientation_vectors(row):
    w, x, y, z = row['qw'], row['qx'], row['qy'], row['qz']

    # R * [1, 0, 0]^T (Forward Vector X)
    vx_x = 1 - 2*(y**2 + z**2)
    vx_y = 2*(x*y + w*z)
    vx_z = 2*(x*z - w*y)
    
    # R * [0, 1, 0]^T (Right Vector Y)
    vy_x = 2*(x*y - w*z)
    vy_y = 1 - 2*(x**2 + z**2)
    vy_z = 2*(y*z + w*x)
    
    # R * [0, 0, 1]^T (Right Vector Y)
    vz_x = 2*(x*z + w*y)
    vz_y = 2*(y*z - w*x)
    vz_z = 1 - 2*x**2 - 2*y**2

    return (vx_x, vx_y, vx_z), (vy_x, vy_y, vy_z), (vz_x, vz_y, vz_z)

# update_animation_frame function
# Updates the position of the lines and markers based on the current frame
def update_animation_frame(frame, data, line, drone_body, arm_x, arm_y, arm_z, arm_length, string_line, egg_marker):
    # Current state
    current = data.iloc[frame]
    
    # Update path history
    history = data.iloc[:frame+1]
    line.set_data(history['x'], history['y'])
    line.set_3d_properties(history['z'])
    
    # Update drone center
    drone_pos = np.array([current['x'], current['y'], current['z']])
    drone_body.set_data([current['x']], [current['y']]) 
    drone_body.set_3d_properties([current['z']])
    
    # Update orientation arms
    vec_x, vec_y, vec_z = get_orientation_vectors(current)
    
    # Draw Forward X arm
    arm_x.set_data([current['x'], current['x'] + vec_x[0]*arm_length], 
                   [current['y'], current['y'] + vec_x[1]*arm_length])
    arm_x.set_3d_properties([current['z'], current['z'] + vec_x[2]*arm_length])
    
    # Draw Right Y arm
    arm_y.set_data([current['x'], current['x'] + vec_y[0]*arm_length], 
                   [current['y'], current['y'] + vec_y[1]*arm_length])
    arm_y.set_3d_properties([current['z'], current['z'] + vec_y[2]*arm_length])

    # Draw Right Y arm
    arm_z.set_data([current['x'], current['x'] + vec_z[0]*arm_length], 
                   [current['y'], current['y'] + vec_z[1]*arm_length])
    arm_z.set_3d_properties([current['z'], current['z'] + vec_z[2]*arm_length])
    
    # Payload animation
    l = current['payload_l']
    theta = current['payload_theta']
    phi = current['payload_phi']
    status = current['payload_status']
    
    # Calculate anchor point using the orientation vectors
    anchor_body = np.array([current['anchor_x'], current['anchor_y'], current['anchor_z']])
    anchor_offset = (anchor_body[0] * np.array(vec_x) + 
                     anchor_body[1] * np.array(vec_y) + 
                     anchor_body[2] * np.array(vec_z))
    anchor_inertial = drone_pos + anchor_offset
    
    if status == "STOWED": # String is hidden
            egg_marker.set_data([anchor_inertial[0]], [anchor_inertial[1]])
            egg_marker.set_3d_properties([anchor_inertial[2]])
            
            string_line.set_data([], [])
            string_line.set_3d_properties([])
            
    else: # LOWERING, FREEFALL, or DROPPED
        egg_x = current['payload_x']
        egg_y = current['payload_y']
        egg_z = current['payload_z']
        
        # Update the egg marker
        egg_marker.set_data([egg_x], [egg_y])
        egg_marker.set_3d_properties([egg_z])
        
        # Hide string if detached
        if status in ["FREEFALL", "DROPPED"]:
            string_line.set_data([], [])
            string_line.set_3d_properties([])
        else:
            string_line.set_data([anchor_inertial[0], egg_x], [anchor_inertial[1], egg_y])
            string_line.set_3d_properties([anchor_inertial[2], egg_z])

    return line, drone_body, arm_x, arm_y, arm_z, string_line, egg_marker

# ----------------------------------------------------------------------
# Main utilities 
# ----------------------------------------------------------------------
# export_simulation_data
# Saves simulation data and parameters with a timestamp
def export_simulation_data(dataframe, params, base_folder="sim_results"):
    # Create output directory if it doesn't exist
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        print(f"[Export] Created directory: {base_folder}")

    # Write timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Export data as csv
    csv_filename = os.path.join(base_folder, f"log_{timestamp}.csv")
    dataframe.to_csv(csv_filename, index=False)
    
    # Export metadata as JSON
    json_filename = os.path.join(base_folder, f"params_{timestamp}.json")
    # Convert any numpy arrays in params to lists for JSON serialization
    safe_params = _sanitize_params_for_json(params)
    
    with open(json_filename, 'w') as f:
        json.dump(safe_params, f, indent=4)

    print(f"\n--- Export Complete ---")
    print(f"Data:   {csv_filename}")
    print(f"Params: {json_filename}")

# plot_simulation_results function
# Generates a 6 panel dashboard of flight data
def plot_simulation_results(df, max_thrust_limit=None):
    print("Processing data for plotting...")
    # Calculate Euler angles (degrees)
    df[['roll', 'pitch', 'yaw']] = df.apply(quat_to_euler, axis=1)
    
    # Create temporary columns for body rates in degrees/s
    df['p_deg'] = np.degrees(df['p'])
    df['q_deg'] = np.degrees(df['q'])
    df['r_deg'] = np.degrees(df['r'])
    
    # Setup figure
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    duration = df['time'].iloc[-1]
    fig.suptitle(f'Flight Data Analysis (Duration: {duration:.2f}s)', fontsize=12)
    
    # Position
    axs[0, 0].plot(df['time'], df['x'], label='X')
    axs[0, 0].plot(df['time'], df['y'], label='Y')
    axs[0, 0].plot(df['time'], df['z'], label='Z', linewidth=2)
    axs[0, 0].set_title('Position (Inertial)', fontsize='small')
    axs[0, 0].set_ylabel('Meters', fontsize='small')
    axs[0, 0].set_xlabel('Time (s)', fontsize='small')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Velocity
    axs[0, 1].plot(df['time'], df['vx'], label='Vx')
    axs[0, 1].plot(df['time'], df['vy'], label='Vy')
    axs[0, 1].plot(df['time'], df['vz'], label='Vz')
    axs[0, 1].set_title('Velocity (Inertial)', fontsize='small')
    axs[0, 1].set_ylabel('m/s', fontsize='small')
    axs[0, 1].set_xlabel('Time (s)', fontsize='small')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Attitude
    axs[1, 0].plot(df['time'], df['roll'], label='Roll')
    axs[1, 0].plot(df['time'], df['pitch'], label='Pitch')
    axs[1, 0].plot(df['time'], df['yaw'], label='Yaw')
    axs[1, 0].set_title('Attitude (Euler Angles)', fontsize='small')
    axs[1, 0].set_ylabel('Degrees', fontsize='small')
    axs[1, 0].set_xlabel('Time (s)', fontsize='small')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Angular Rates
    axs[1, 1].plot(df['time'], df['p_deg'], label='P (Roll)')
    axs[1, 1].plot(df['time'], df['q_deg'], label='Q (Pitch)')
    axs[1, 1].plot(df['time'], df['r_deg'], label='R (Yaw)')
    axs[1, 1].set_title('Body Rates', fontsize='small')
    axs[1, 1].set_ylabel('deg/s', fontsize='small')
    axs[1, 1].set_xlabel('Time (s)', fontsize='small')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    # Motor Thrusts
    axs[2, 0].plot(df['time'], df['thrust_m1'], label='M1', alpha=0.8)
    axs[2, 0].plot(df['time'], df['thrust_m2'], label='M2', alpha=0.8)
    axs[2, 0].plot(df['time'], df['thrust_m3'], label='M3', alpha=0.8)
    axs[2, 0].plot(df['time'], df['thrust_m4'], label='M4', alpha=0.8)
    if max_thrust_limit:
        axs[2, 0].axhline(max_thrust_limit, color='r', linestyle='--', label='Limit')
    axs[2, 0].set_title('Motor Thrusts', fontsize='small')
    axs[2, 0].set_ylabel('Newtons', fontsize='small')
    axs[2, 0].set_xlabel('Time (s)', fontsize='small')
    axs[2, 0].grid(True)
    axs[2, 0].legend(loc='upper right', fontsize='small')
    
    # Ground Track
    axs[2, 1].plot(df['x'], df['y'], 'b-', label='Path')
    axs[2, 1].plot(df['x'].iloc[0], df['y'].iloc[0], 'go', label='Start')
    axs[2, 1].plot(df['x'].iloc[-1], df['y'].iloc[-1], 'rx', label='End')
    axs[2, 1].set_title('Ground Track (Top Down)', fontsize='small')
    axs[2, 1].set_xlabel('X (m)', fontsize='small')
    axs[2, 1].set_ylabel('Y (m)', fontsize='small')
    axs[2, 1].axis('equal')
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    att_fig = plt.figure()
    # Attitude
    plt.plot(df['time'], df['roll'], label='Roll')
    plt.plot(df['time'], df['pitch'], label='Pitch')
    plt.plot(df['time'], df['yaw'], label='Yaw')
    plt.title('Attitude (Euler Angles)')
    plt.ylabel('Degrees')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    
    # Angular Rates
    rates_fig = plt.figure()
    plt.plot(df['time'], df['p_deg'], label='P (Roll)')
    plt.plot(df['time'], df['q_deg'], label='Q (Pitch)')
    plt.plot(df['time'], df['r_deg'], label='R (Yaw)')
    plt.title('Body Rates')
    plt.ylabel('deg/s')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()

    pos_fig = plt.figure()
    # Position
    plt.plot(df['time'], df['x'], label='X')
    plt.plot(df['time'], df['x_des'], label='X Setpoint')
    plt.plot(df['time'], df['y'], label='Y')
    plt.plot(df['time'], df['y_des'], label='Y Setpoint')
    plt.plot(df['time'], df['z'], label='Z', linewidth=2)
    plt.plot(df['time'], df['z_des'], label='Z Setpoint')
    plt.title('Position (Inertial)')
    plt.ylabel('Meters')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    
    vel_fig = plt.figure()
    # Velocity
    plt.plot(df['time'], df['vx'], label='Vx')
    plt.plot(df['time'], df['vy'], label='Vy')
    plt.plot(df['time'], df['vz'], label='Vz')
    plt.title('Velocity (Inertial)')
    plt.ylabel('m/s')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()

    thrust_fig = plt.figure()
    plt.plot(df['time'], df['thrust_m1'], label='M1', alpha=0.8)
    plt.plot(df['time'], df['thrust_m2'], label='M2', alpha=0.8)
    plt.plot(df['time'], df['thrust_m3'], label='M3', alpha=0.8)
    plt.plot(df['time'], df['thrust_m4'], label='M4', alpha=0.8)
    if max_thrust_limit:
        plt.axhline(max_thrust_limit, color='r', linestyle='--', label='Limit')
    plt.title('Motor Thrusts', fontsize='small')
    plt.ylabel('Newtons', fontsize='small')
    plt.xlabel('Time (s)', fontsize='small')
    plt.grid(True)
    plt.title('Motor Thrusts')
    plt.ylabel('Thrust (N)')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.show()
    
# animate_simulation function
# Creates a 3D animation of the drone's flight path and orientation
def animate_simulation_3d(df, target_trajectory=None, filename=None, waypoints=None):
    print("Generating 3D Animation...")
    
    # Downsample data
    skip = 5
    data = df.iloc[::skip].reset_index(drop=True)
    
    # Setup figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    
    margin = 1.0
    ax.set_xlim(data['x'].min()-margin, data['x'].max()+margin)
    ax.set_ylim(data['y'].min()-margin, data['y'].max()+margin)
    ax.set_zlim(0, data['z'].max()+margin)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Drone Flight Replay (Speed: {skip}x)')
    
    # Static elements
    # Ground
    xx, yy = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10), np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10))
    ax.plot_surface(xx, yy, xx*0, color='gray', alpha=0.2)
    
    ax.view_init(elev=15., azim=45)
    
    # Target path
    if target_trajectory is not None:
        tx, ty, tz = target_trajectory[0], target_trajectory[1], target_trajectory[2]
        ax.plot(tx, ty, tz, 'r--', label='Target Path', linewidth=1)
        
    # Waypoints
    if waypoints is not None:
        for label, point in waypoints:
            x, y, z = point['pos']
            ax.scatter(x, y, z, color=point['color'], marker=point.get('marker', 'o'), s=point.get('size', 80), zorder=5)
            ax.text(x, y, z, f'  {label}', color=point['color'], fontsize=8)

    # Dynamic elements (initialized empty)
    line, = ax.plot([], [], [], 'b-', linewidth=1, label='Actual Path')
    drone_body, = ax.plot([], [], [], 'ko', markersize=5)
    arm_x, = ax.plot([], [], [], 'r-', linewidth=2) 
    arm_y, = ax.plot([], [], [], 'g-', linewidth=2) 
    arm_z, = ax.plot([], [], [], 'b-', linewidth=2) 
    string_line, = ax.plot([], [], [], 'k-', linewidth=1, alpha=0.6)
    egg_marker, = ax.plot([], [], [], 'mo', markersize=6, label='Egg')

    # Create animation
    update_func = partial(update_animation_frame, data=data, line=line, drone_body=drone_body, arm_x=arm_x, arm_y=arm_y, arm_z=arm_z, arm_length=0.5, string_line=string_line, egg_marker=egg_marker)
    ani = animation.FuncAnimation(fig, update_func, frames=len(data), interval=30, blit=False)
    
    if filename:
        print(f"Saving animation to {filename}...")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ani.save(filename, writer='ffmpeg', fps=30)
        
    plt.legend()
    plt.show()
    
# ----------------------------------------------------------------------
# FLIGHT DATA COMPARISON
# ----------------------------------------------------------------------
# Column maps - set a value to None if that data isn't available
OUTER_LOOP_COL_MAP = {
    "time": "t",
    "x": "x",  "y": "y",  "z": "z",
    "xd": "xd",  "yd": "yd",  "zd": "zd",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "vdx": "vdx", "vdy": "vdy", "vdz": "vdz",
    "adx": "adx", "ady": "ady", "adz": "adz",
    "jdx": "jdx", "jdy": "jdy", "jdz": "jdz",
    "throttle": "throttle"
}

INNER_LOOP_COL_MAP = {
    "time": "t",
    "qw": "qw", "qx": "qx", "qy": "qy", "qz": "qz",
    "qwd": "qwd", "qxd": "qxd", "qyd": "qyd", "qzd": "qzd",
    "p": "wx", "q": "wy", "r": "wz", # If direct body rate columns are available point at p, q, r and set qd* entries to None
    "pd": "wdx", "qd": "wdy", "rd": "wdz",
}

# get_col function
# Returns a Series from df using col_map, or None if the column is unavailable
def get_col(df, col_map, key):
    col = col_map.get(key)
    if col is None or col not in df.columns:
        return None
    
    return df[col]

# sim_col function
# Returns a numpy array for a column from the sim DataFrame, or None if missing
def sim_col(sim_df, col):
    if col not in sim_df.columns:
        return None
    
    return sim_df[col].values

# actual_col function
# Returns a numpy array for a column from an actual flight DataFrame, or None if missing
def actual_col(df, col):
    if df is None or col not in df.columns:
        return None
    
    return df[col].values

# align_actual_time function
# Shifts actual flight timestamps to start at the same point as the sim, plus any manual offset
def align_actual_time(df, col_map, sim_t, time_offset):
    t_col = col_map.get("time")
    if df is None or t_col not in df.columns:
        return None
    
    return df[t_col].values + (sim_t[0] - df[t_col].values[0]) + time_offset

# draw_comparison_subplots function
# Draws sim vs actual traces onto an existing row of axes given a list of subplot specs
def draw_comparison_subplots(axs, sim_t, subplot_specs):
    for ax, (title, ylabel, sim_s, actual_t, actual_s) in zip(axs, subplot_specs):
        if sim_s is not None:
            ax.plot(sim_t, sim_s, label="Sim", linewidth=1.5)
            
        if actual_s is not None and actual_t is not None:
            ax.plot(actual_t, actual_s, label="Actual", linewidth=1.2, linestyle="--", alpha=0.85)
            
        ax.set_title(title, fontsize="small")
        ax.set_ylabel(ylabel, fontsize="small")
        ax.set_xlabel("Time (s)", fontsize="small")
        ax.grid(True)
        ax.legend(fontsize="x-small")

def draw_flight_tracking_comparison_subplots(axs, sim_t, subplot_specs):
    for ax, (title, ylabel, sim_s, actual_t, actual_s) in zip(axs, subplot_specs):
        if sim_s is not None:
            ax.plot(sim_t, sim_s, label="Actual", linewidth=1.5)
            
        if actual_s is not None and actual_t is not None:
            ax.plot(actual_t, actual_s, label="Desired", linewidth=1.2, linestyle="--", alpha=0.85)
            
        ax.set_title(title, fontsize="small")
        ax.set_ylabel(ylabel, fontsize="small")
        ax.set_xlabel("Time (s)", fontsize="small")
        ax.grid(True)
        ax.legend(fontsize="x-small")

# body_rates_from_qd function
# Derives body rates p/q/r (rad/s) from quaternion derivatives
def body_rates_from_qd(df, col_map):
    needed = ["qw", "qx", "qy", "qz", "qdw", "qdx", "qdy", "qdz"]
    cols = {k: get_col(df, col_map, k) for k in needed}
    
    if any(v is None for v in cols.values()):
        return None
    
    qw  = cols["qw"].values;  qx  = cols["qx"].values
    qy  = cols["qy"].values;  qz  = cols["qz"].values
    qdw = cols["qdw"].values; qdx = cols["qdx"].values
    qdy = cols["qdy"].values; qdz = cols["qdz"].values
    
    p = 2 * (qw*qdx - qx*qdw - qy*qdz + qz*qdy)
    q = 2 * (qw*qdy + qx*qdz - qy*qdw - qz*qdx)
    r = 2 * (qw*qdz - qx*qdy + qy*qdx - qz*qdw)
    
    return pd.DataFrame({"p": p, "q": q, "r": r})

# load_and_normalise_csv function
# Reads a CSV and re-zeros the time column from epoch to elapsed seconds if needed
def load_and_normalise_csv(path, col_map, time_is_epoch):
    if path is None:
        return None
    
    df = pd.read_csv(path)
    t_col = col_map.get("time")
    
    if t_col and t_col in df.columns and time_is_epoch:
        df[t_col] = df[t_col] - df[t_col].iloc[0]
        
    return df

# load_flight_data function
# Loads outer-loop (position/velocity) and inner-loop (attitude/rates) CSVs and returns normalised DataFrames
def load_flight_data(outer_csv=None, inner_csv=None, outer_col_map=None, inner_col_map=None, time_is_epoch=True):
    ocm = outer_col_map or OUTER_LOOP_COL_MAP
    icm = inner_col_map or INNER_LOOP_COL_MAP

    outer = load_and_normalise_csv(outer_csv, ocm, time_is_epoch)
    inner = load_and_normalise_csv(inner_csv, icm, time_is_epoch)

    if inner is not None:
        qcols = {k: get_col(inner, icm, k) for k in ["qw", "qx", "qy", "qz"]}
        if all(v is not None for v in qcols.values()):
            tmp = pd.DataFrame({k: v.values for k, v in qcols.items()})
            inner[["roll", "pitch", "yaw"]] = tmp.apply(quat_to_euler, axis=1)

        rates = body_rates_from_qd(inner, icm)
        if rates is not None:
            inner[["p", "q", "r"]] = rates

    return {"outer": outer, "inner": inner, "ocm": ocm, "icm": icm}

# plot_sim_vs_actual function
# Plots position, velocity, attitude, and body rate comparison between flight data and simulation prediction
def plot_sim_vs_actual(sim_df, flight_data, inner_time_offset = 0.0, outer_time_offset=0.0, t_start=0.0, t_end=None):
    outer = flight_data["outer"]
    inner = flight_data["inner"]
    ocm   = flight_data["ocm"]
    icm   = flight_data["icm"]
    sim_t = sim_df["time"].values

    ot = align_actual_time(outer, ocm, sim_t, outer_time_offset)
    it = align_actual_time(inner, icm, sim_t, inner_time_offset)
    
    t_end = t_end if t_end is not None else sim_t[-1]

    outer_mask = (ot >= t_start) & (ot <= t_end)
    inner_mask = (it >= t_start) & (it <= t_end)
    sim_mask   = (sim_t >= t_start) & (sim_t <= t_end)

    outer  = outer[outer_mask].reset_index(drop=True) if outer is not None else None
    inner  = inner[inner_mask].reset_index(drop=True) if inner is not None else None
    ot     = ot[outer_mask]
    it     = it[inner_mask]
    sim_t  = sim_t[sim_mask]
    sim_df = sim_df[sim_mask].reset_index(drop=True)
    
    # Euler angles from sim quaternions
    sim_euler = None
    if all(c in sim_df.columns for c in ["qw", "qx", "qy", "qz"]):
        sim_euler = sim_df[["qw", "qx", "qy", "qz"]].apply(quat_to_euler, axis=1)
        sim_euler.columns = ["roll", "pitch", "yaw"]

    # Position
    pos_fig, pos_axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    pos_fig.suptitle("Position: Sim vs Actual", fontsize=11)
    draw_comparison_subplots(pos_axs, sim_t, [
        ("X", "m",   sim_col(sim_df, "x"),  ot, get_col(outer, ocm, "x").values  if get_col(outer, ocm, "x")  is not None else None),
        ("Y", "m",   sim_col(sim_df, "y"),  ot, get_col(outer, ocm, "y").values  if get_col(outer, ocm, "y")  is not None else None),
        ("Z", "m",   sim_col(sim_df, "z"),  ot, get_col(outer, ocm, "z").values  if get_col(outer, ocm, "z")  is not None else None)
    ])

    # Velocity
    vel_fig, vel_axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    vel_fig.suptitle("Velocity: Sim vs Actual", fontsize=11)
    draw_comparison_subplots(vel_axs, sim_t, [
        ("Vx", "m/s", sim_col(sim_df, "vx"), ot, actual_col(outer, "vx")),
        ("Vy", "m/s", sim_col(sim_df, "vy"), ot, actual_col(outer, "vy")),
        ("Vz", "m/s", sim_col(sim_df, "vz"), ot, actual_col(outer, "vz")),
    ])

    # Attitude
    att_fig, att_axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    att_fig.suptitle("Attitude: Sim vs Actual", fontsize=11)
    draw_comparison_subplots(att_axs, sim_t, [
        ("Roll",  "deg", sim_euler["roll"].values  if sim_euler is not None else None, it, actual_col(inner, "roll")),
        ("Pitch", "deg", sim_euler["pitch"].values if sim_euler is not None else None, it, actual_col(inner, "pitch")),
        ("Yaw",   "deg", sim_euler["yaw"].values   if sim_euler is not None else None, it, actual_col(inner, "yaw")),
    ])

    # Body rates
    sim_p = np.degrees(sim_col(sim_df, "p")) if sim_col(sim_df, "p") is not None else None
    sim_q = np.degrees(sim_col(sim_df, "q")) if sim_col(sim_df, "q") is not None else None
    sim_r = np.degrees(sim_col(sim_df, "r")) if sim_col(sim_df, "r") is not None else None
    act_p = np.degrees(get_col(inner, icm, "p").values) if get_col(inner, icm, "p") is not None else None
    act_q = np.degrees(get_col(inner, icm, "q").values) if get_col(inner, icm, "q") is not None else None
    act_r = np.degrees(get_col(inner, icm, "r").values) if get_col(inner, icm, "r") is not None else None
    rates_fig, rates_axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    rates_fig.suptitle("Body Rates: Sim vs Actual", fontsize=11)
    
    draw_comparison_subplots(rates_axs, sim_t, [
        ("P (Roll)",  "deg/s", sim_p, it, act_p),
        ("Q (Pitch)", "deg/s", sim_q, it, act_q),
        ("R (Yaw)",   "deg/s", sim_r, it, act_r),
    ])


    # Position
    pos_tracking_fig, pos_tracking_axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    pos_tracking_fig.suptitle("Flight Data Position: Actual vs Desired", fontsize=11)
    draw_flight_tracking_comparison_subplots(pos_tracking_axs, ot, [
        ("X", "m",   get_col(outer, ocm, "x").values,  ot, get_col(outer, ocm, "xd").values  if get_col(outer, ocm, "xd")  is not None else None),
        ("Y", "m",   get_col(outer, ocm, "y").values,  ot, get_col(outer, ocm, "yd").values  if get_col(outer, ocm, "yd")  is not None else None),
        ("Z", "m",   get_col(outer, ocm, "z").values,  ot, get_col(outer, ocm, "zd").values  if get_col(outer, ocm, "zd")  is not None else None)
    ])

    # Velocity
    vel_tracking_fig, vel_tracking_axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    vel_tracking_fig.suptitle("Flight Data Velocity: Actual vs Desired", fontsize=11)
    draw_flight_tracking_comparison_subplots(vel_tracking_axs, ot, [
        ("Vx", "m/s", actual_col(outer, "vx"), ot, actual_col(outer, "vxd")),
        ("Vy", "m/s", actual_col(outer, "vy"), ot, actual_col(outer, "vyd")),
        ("Vz", "m/s", actual_col(outer, "vz"), ot, actual_col(outer, "vzd")),
    ])

    # Attitude
    attitude_tracking_fig, attitude_tracking_axs = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    attitude_tracking_fig.suptitle("Flight Data Attitude: Actual vs Desired", fontsize=11)
    draw_flight_tracking_comparison_subplots(attitude_tracking_axs, it, [
        ("qw", "", actual_col(inner, "qw"), it, actual_col(inner, "qwd")),
        ("qx", "", actual_col(inner, "qx"), it, actual_col(inner, "qxd")),
        ("qy", "", actual_col(inner, "qy"), it, actual_col(inner, "qyd")),
        ("qz", "", actual_col(inner, "qz"), it, actual_col(inner, "qzd")),
    ])

    # Body rates
    br_tracking_fig, br_tracking_axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    br_tracking_fig.suptitle("Flight Data Body Rates: Actual vs Desired", fontsize=11)
    draw_flight_tracking_comparison_subplots(br_tracking_axs, it, [
        ("wx", "rad/s", actual_col(inner, "wx"), it, actual_col(inner, "wxd")),
        ("wy", "rad/s", actual_col(inner, "wy"), it, actual_col(inner, "wyd")),
        ("wz", "rad/s", actual_col(inner, "wz"), it, actual_col(inner, "wzd")),
    ])

    plt.show()
    
# update_comparison_frame function
# Updates sim drone and actual drone body and arms for each animation frame
def update_comparison_frame(frame, sim_data, actual_data, sim_line, actual_line, sim_body, actual_body, sim_arm_x, sim_arm_y, sim_arm_z, act_arm_x, act_arm_y, act_arm_z, arm_length, string_line, egg_marker):
    sim_cur    = sim_data.iloc[frame]
    actual_cur = actual_data.iloc[frame]

    # Sim path history
    sim_hist = sim_data.iloc[:frame+1]
    sim_line.set_data(sim_hist['x'], sim_hist['y'])
    sim_line.set_3d_properties(sim_hist['z'])

    # Actual path history
    act_hist = actual_data.iloc[:frame+1]
    actual_line.set_data(act_hist['x'], act_hist['y'])
    actual_line.set_3d_properties(act_hist['z'])

    # Sim drone body
    sim_body.set_data([sim_cur['x']], [sim_cur['y']])
    sim_body.set_3d_properties([sim_cur['z']])

    # Actual drone body
    actual_body.set_data([actual_cur['x']], [actual_cur['y']])
    actual_body.set_3d_properties([actual_cur['z']])

    # Sim orientation arms
    sim_pos = np.array([sim_cur['x'], sim_cur['y'], sim_cur['z']])
    vec_x, vec_y, vec_z = get_orientation_vectors(sim_cur)

    sim_arm_x.set_data([sim_pos[0], sim_pos[0] + vec_x[0]*arm_length], [sim_pos[1], sim_pos[1] + vec_x[1]*arm_length])
    sim_arm_x.set_3d_properties([sim_pos[2], sim_pos[2] + vec_x[2]*arm_length])
    sim_arm_y.set_data([sim_pos[0], sim_pos[0] + vec_y[0]*arm_length], [sim_pos[1], sim_pos[1] + vec_y[1]*arm_length])
    sim_arm_y.set_3d_properties([sim_pos[2], sim_pos[2] + vec_y[2]*arm_length])
    sim_arm_z.set_data([sim_pos[0], sim_pos[0] + vec_z[0]*arm_length], [sim_pos[1], sim_pos[1] + vec_z[1]*arm_length])
    sim_arm_z.set_3d_properties([sim_pos[2], sim_pos[2] + vec_z[2]*arm_length])

    # Actual orientation arms
    act_pos = np.array([actual_cur['x'], actual_cur['y'], actual_cur['z']])
    act_vx, act_vy, act_vz = get_orientation_vectors(actual_cur)

    act_arm_x.set_data([act_pos[0], act_pos[0] + act_vx[0]*arm_length], [act_pos[1], act_pos[1] + act_vx[1]*arm_length])
    act_arm_x.set_3d_properties([act_pos[2], act_pos[2] + act_vx[2]*arm_length])
    act_arm_y.set_data([act_pos[0], act_pos[0] + act_vy[0]*arm_length], [act_pos[1], act_pos[1] + act_vy[1]*arm_length])
    act_arm_y.set_3d_properties([act_pos[2], act_pos[2] + act_vy[2]*arm_length])
    act_arm_z.set_data([act_pos[0], act_pos[0] + act_vz[0]*arm_length], [act_pos[1], act_pos[1] + act_vz[1]*arm_length])
    act_arm_z.set_3d_properties([act_pos[2], act_pos[2] + act_vz[2]*arm_length])

    # Sim payload
    l      = sim_cur['payload_l']
    status = sim_cur['payload_status']
    anchor_body   = np.array([sim_cur['anchor_x'], sim_cur['anchor_y'], sim_cur['anchor_z']])
    anchor_offset = (anchor_body[0] * np.array(vec_x) +
                     anchor_body[1] * np.array(vec_y) +
                     anchor_body[2] * np.array(vec_z))
    anchor_inertial = sim_pos + anchor_offset

    if status == "STOWED":
        egg_marker.set_data([anchor_inertial[0]], [anchor_inertial[1]])
        egg_marker.set_3d_properties([anchor_inertial[2]])
        string_line.set_data([], [])
        string_line.set_3d_properties([])
    else:
        egg_x, egg_y, egg_z = sim_cur['payload_x'], sim_cur['payload_y'], sim_cur['payload_z']
        egg_marker.set_data([egg_x], [egg_y])
        egg_marker.set_3d_properties([egg_z])
        if status in ["FREEFALL", "DROPPED"]:
            string_line.set_data([], [])
            string_line.set_3d_properties([])
        else:
            string_line.set_data([anchor_inertial[0], egg_x], [anchor_inertial[1], egg_y])
            string_line.set_3d_properties([anchor_inertial[2], egg_z])

    return (sim_line, actual_line, sim_body, actual_body,
            sim_arm_x, sim_arm_y, sim_arm_z,
            act_arm_x, act_arm_y, act_arm_z,
            string_line, egg_marker)

# animate_sim_vs_actual function
# Animates the simulated drone alongside the actual drone on a shared 3D axes
def animate_sim_vs_actual(sim_df, flight_data, target_trajectory=None, inner_time_offset = 0.0, outer_time_offset=0.0, t_start=0.0, t_end=None, filename=None, waypoints=None):
    print("Generating Sim vs Actual 3D Animation...")

    outer = flight_data["outer"]
    inner = flight_data["inner"]
    ocm   = flight_data["ocm"]
    icm   = flight_data["icm"]

    # Build aligned time arrays
    sim_t  = sim_df["time"].values
    ot     = align_actual_time(outer, ocm, sim_t, outer_time_offset)
    it     = align_actual_time(inner, icm, sim_t, inner_time_offset)

    # Trim to window
    t_end = t_end if t_end is not None else sim_t[-1]

    sim_mask   = (sim_t >= t_start) & (sim_t <= t_end)
    outer_mask = (ot    >= t_start) & (ot    <= t_end)
    inner_mask = (it    >= t_start) & (it    <= t_end)

    sim_df = sim_df[sim_mask].reset_index(drop=True)
    outer  = outer[outer_mask].reset_index(drop=True)
    inner  = inner[inner_mask].reset_index(drop=True)
    sim_t  = sim_t[sim_mask]
    ot     = ot[outer_mask]
    it     = it[inner_mask]

    # Interpolate actual position and attitude onto a common time axis (sim_t)
    outer_t_col = ocm.get("time")
    inner_t_col = icm.get("time")

    actual_x  = np.interp(sim_t, ot,  outer["x"].values)
    actual_y  = np.interp(sim_t, ot,  outer["y"].values)
    actual_z  = np.interp(sim_t, ot,  outer["z"].values)
    actual_qw = np.interp(sim_t, it,  inner["qw"].values)
    actual_qx = np.interp(sim_t, it,  inner["qx"].values)
    actual_qy = np.interp(sim_t, it,  inner["qy"].values)
    actual_qz = np.interp(sim_t, it,  inner["qz"].values)

    actual_data = pd.DataFrame({
        "x": actual_x, "y": actual_y, "z": actual_z,
        "qw": actual_qw, "qx": actual_qx, "qy": actual_qy, "qz": actual_qz,
    })

    # Downsample
    skip     = 5
    sim_data    = sim_df.iloc[::skip].reset_index(drop=True)
    actual_data = actual_data.iloc[::skip].reset_index(drop=True)

    # Axis limits from both datasets combined
    all_x = np.concatenate([sim_data['x'].values, actual_data['x'].values])
    all_y = np.concatenate([sim_data['y'].values, actual_data['y'].values])
    all_z = np.concatenate([sim_data['z'].values, actual_data['z'].values])
    margin = 1.0

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.set_xlim(all_x.min()-margin, all_x.max()+margin)
    ax.set_ylim(all_y.min()-margin, all_y.max()+margin)
    ax.set_zlim(0, all_z.max()+margin)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Sim vs Actual Flight Replay (Speed: {skip}x)')
    ax.view_init(elev=15., azim=45)

    # Ground plane
    xx, yy = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10),
                         np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10))
    ax.plot_surface(xx, yy, xx*0, color='gray', alpha=0.2)
    
    # Target path
    if target_trajectory is not None:
        tx, ty, tz = target_trajectory[0], target_trajectory[1], target_trajectory[2]
        ax.plot(tx, ty, tz, 'r--', label='Target Path', linewidth=1)
        
    # Waypoints
    if waypoints is not None:
        for label, point in waypoints:
            x, y, z = point['pos']
            ax.scatter(x, y, z, color=point['color'], marker=point.get('marker', 'o'), s=point.get('size', 80), zorder=5)
            ax.text(x, y, z, f'  {label}', color=point['color'], fontsize=8)

    # Dynamic elements — sim in blue, actual in orange
    sim_line,    = ax.plot([], [], [], 'b-',  linewidth=1,  label='Sim Path')
    actual_line, = ax.plot([], [], [], '-',   linewidth=1,  label='Actual Path', color='orange')
    sim_body,    = ax.plot([], [], [], 'bo',  markersize=5)
    actual_body, = ax.plot([], [], [], 'o',   markersize=5, color='orange')
    sim_arm_x,   = ax.plot([], [], [], 'r-',  linewidth=2)
    sim_arm_y,   = ax.plot([], [], [], 'g-',  linewidth=2)
    sim_arm_z,   = ax.plot([], [], [], 'b-',  linewidth=2)
    act_arm_x,   = ax.plot([], [], [], '-',   linewidth=2, color='#ff6666')
    act_arm_y,   = ax.plot([], [], [], '-',   linewidth=2, color='#66cc66')
    act_arm_z,   = ax.plot([], [], [], '-',   linewidth=2, color='#ffaa00')
    string_line, = ax.plot([], [], [], 'k-',  linewidth=1, alpha=0.6)
    egg_marker,  = ax.plot([], [], [], 'mo',  markersize=6, label='Egg')
    
    sim_line.set_alpha(0.4)
    sim_body.set_alpha(0.4)
    sim_arm_x.set_alpha(0.4)
    sim_arm_y.set_alpha(0.4)
    sim_arm_z.set_alpha(0.4)
    egg_marker.set_alpha(0.4)

    update_func = partial(
        update_comparison_frame,
        sim_data=sim_data, actual_data=actual_data,
        sim_line=sim_line, actual_line=actual_line,
        sim_body=sim_body, actual_body=actual_body,
        sim_arm_x=sim_arm_x, sim_arm_y=sim_arm_y, sim_arm_z=sim_arm_z,
        act_arm_x=act_arm_x, act_arm_y=act_arm_y, act_arm_z=act_arm_z,
        arm_length=0.5, string_line=string_line, egg_marker=egg_marker,
    )

    ani = animation.FuncAnimation(fig, update_func, frames=len(sim_data), interval=30, blit=False)

    if filename:
        print(f"Saving animation to {filename}...")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ani.save(filename, writer='ffmpeg', fps=30)

    plt.legend()
    plt.show()