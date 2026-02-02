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
    
def quat_multiply(q1, q2):
    w1, x1, y1, z1, w2, x2, y2, z2 = q1[0], q1[1], q1[2], q1[3], q2[0], q2[1], q2[2], q2[3]
    output_quat = np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*x2])
    return output_quat

def quat_conjugate(q1):
    output_quat = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    return output_quat

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
def update_animation_frame(frame, data, line, drone_body, arm_x, arm_y, arm_z, arm_length):
    # Current state
    current = data.iloc[frame]
    
    # Update path history
    history = data.iloc[:frame+1]
    line.set_data(history['x'], history['y'])
    line.set_3d_properties(history['z'])
    
    # Update drone center
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
    
    return line, drone_body, arm_x, arm_y, arm_z

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
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    duration = df['time'].iloc[-1]
    fig.suptitle(f'Flight Data Analysis (Duration: {duration:.2f}s)', fontsize=16)
    
    # Position
    axs[0, 0].plot(df['time'], df['x'], label='X')
    axs[0, 0].plot(df['time'], df['y'], label='Y')
    axs[0, 0].plot(df['time'], df['z'], label='Z', linewidth=2)
    axs[0, 0].set_title('Position (Inertial)')
    axs[0, 0].set_ylabel('Meters')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Velocity
    axs[0, 1].plot(df['time'], df['vx'], label='Vx')
    axs[0, 1].plot(df['time'], df['vy'], label='Vy')
    axs[0, 1].plot(df['time'], df['vz'], label='Vz')
    axs[0, 1].set_title('Velocity (Inertial)')
    axs[0, 1].set_ylabel('m/s')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Attitude
    axs[1, 0].plot(df['time'], df['roll'], label='Roll')
    axs[1, 0].plot(df['time'], df['pitch'], label='Pitch')
    axs[1, 0].plot(df['time'], df['yaw'], label='Yaw')
    axs[1, 0].set_title('Attitude (Euler Angles)')
    axs[1, 0].set_ylabel('Degrees')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Angular Rates
    axs[1, 1].plot(df['time'], df['p_deg'], label='P (Roll)')
    axs[1, 1].plot(df['time'], df['q_deg'], label='Q (Pitch)')
    axs[1, 1].plot(df['time'], df['r_deg'], label='R (Yaw)')
    axs[1, 1].set_title('Body Rates')
    axs[1, 1].set_ylabel('deg/s')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    # Motor Thrusts
    axs[2, 0].plot(df['time'], df['thrust_m1'], label='M1', alpha=0.8)
    axs[2, 0].plot(df['time'], df['thrust_m2'], label='M2', alpha=0.8)
    axs[2, 0].plot(df['time'], df['thrust_m3'], label='M3', alpha=0.8)
    axs[2, 0].plot(df['time'], df['thrust_m4'], label='M4', alpha=0.8)
    if max_thrust_limit:
        axs[2, 0].axhline(max_thrust_limit, color='r', linestyle='--', label='Limit')
    axs[2, 0].set_title('Motor Thrusts')
    axs[2, 0].set_ylabel('Newtons')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].grid(True)
    axs[2, 0].legend(loc='upper right', fontsize='small')
    
    # Ground Track
    axs[2, 1].plot(df['x'], df['y'], 'b-', label='Path')
    axs[2, 1].plot(df['x'].iloc[0], df['y'].iloc[0], 'go', label='Start')
    axs[2, 1].plot(df['x'].iloc[-1], df['y'].iloc[-1], 'rx', label='End')
    axs[2, 1].set_title('Ground Track (Top Down)')
    axs[2, 1].set_xlabel('X (m)')
    axs[2, 1].set_ylabel('Y (m)')
    axs[2, 1].axis('equal')
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
# animate_simulation function
# Creates a 3D animation of the drone's flight path and orientation
def animate_simulation_3d(df, target_trajectory=None, filename=None):
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
    
    # Target path
    if target_trajectory is not None:
        '''
        tx, ty, tz = target_trajectory[:,0], target_trajectory[:,1], target_trajectory[:,2]
        ax.plot(tx, ty, tz, 'r--', label='Target Path', linewidth=1)
        '''

    # Dynamic elements (initialized empty)
    line, = ax.plot([], [], [], 'b-', linewidth=1, label='Actual Path')
    drone_body, = ax.plot([], [], [], 'ko', markersize=5)
    arm_x, = ax.plot([], [], [], 'r-', linewidth=2) 
    arm_y, = ax.plot([], [], [], 'g-', linewidth=2) 
    arm_z, = ax.plot([], [], [], 'b-', linewidth=2) 

    # Greate animation
    update_func = partial(update_animation_frame, data=data, line=line, drone_body=drone_body, arm_x=arm_x, arm_y=arm_y, arm_z=arm_z, arm_length=0.5)
    ani = animation.FuncAnimation(fig, update_func, frames=len(data), interval=30, blit=False)
    
    if filename:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='ffmpeg', fps=30)
        
    plt.legend()
    plt.show()