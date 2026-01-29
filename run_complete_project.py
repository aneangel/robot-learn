#!/usr/bin/env python3
"""
================================================================================
ECEN524 Robot Learning Project - Complete Pipeline
Anthony Angeles
February 02, 2026
================================================================================

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DEMO_FOLDER = 'demotrajectories'
OUTPUT_DIR = 'project_output'
EXCLUDED_DEMOS = ['demo6', 'demo10']  # Known problematic demos

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_trajectory(filepath):
    """Load trajectory data from CSV file."""
    timestamps = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(0,), dtype=str)
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7))
    return timestamps, data


def parse_timestamps(timestamps):
    """Convert timestamp strings to relative seconds."""
    times = []
    for ts in timestamps:
        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
        times.append(dt.timestamp())
    times = np.array(times)
    return times - times[0]


def resample_trajectory(times, positions, n_points=200):
    """Resample trajectory to uniform number of points."""
    t_norm = times / times[-1] if times[-1] > 0 else times
    t_uniform = np.linspace(0, 1, n_points)
    
    positions_resampled = np.zeros((n_points, positions.shape[1]))
    for dim in range(positions.shape[1]):
        interp_func = interp1d(t_norm, positions[:, dim], kind='linear', fill_value='extrapolate')
        positions_resampled[:, dim] = interp_func(t_uniform)
    
    return t_uniform, positions_resampled


def print_section(title, char='='):
    """Print section header."""
    print(f"\n{char*70}")
    print(title)
    print(f"{char*70}")


# ============================================================================
# SECTION A: DATA PLOTTING AND POI ANALYSIS
# ============================================================================

def run_section_a(demos, output_dir):
    """
    Section A: Plot demonstration data and identify Points of Interest.
    
    This section:
    - Plots 3D trajectories of all demonstrations
    - Plots position vs time for each axis
    - Identifies pick and place events (POI)
    - Computes velocity profiles
    """
    print_section("SECTION A: DATA PLOTTING AND POI ANALYSIS (10%)")
    
    section_dir = os.path.join(output_dir, 'section_a_plots')
    os.makedirs(section_dir, exist_ok=True)
    
    # 1. Plot all trajectories in 3D
    print("\n1. Creating 3D trajectory overview...")
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(demos)))
    
    for (name, data), color in zip(demos.items(), colors):
        times, trajectory = data
        positions = trajectory[:, :3]
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=color, linewidth=1.5, alpha=0.7, label=name)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('All Demonstration Trajectories (3D)')
    ax1.legend(loc='upper left', fontsize=7, ncol=2)
    
    # XY projection
    ax2 = fig.add_subplot(1, 2, 2)
    for (name, data), color in zip(demos.items(), colors):
        times, trajectory = data
        positions = trajectory[:, :3]
        ax2.plot(positions[:, 0], positions[:, 1], 
                color=color, linewidth=1.5, alpha=0.7, label=name)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('All Trajectories (Top-Down View)')
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, 'all_trajectories_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {section_dir}/all_trajectories_3d.png")
    
    # 2. Position vs Time plots
    print("\n2. Creating position vs time plots...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for (name, data), color in zip(demos.items(), colors):
        times, trajectory = data
        positions = trajectory[:, :3]
        t_norm = times / times[-1]  # Normalize time
        
        axes[0].plot(t_norm, positions[:, 0], color=color, alpha=0.6, linewidth=1, label=name)
        axes[1].plot(t_norm, positions[:, 1], color=color, alpha=0.6, linewidth=1)
        axes[2].plot(t_norm, positions[:, 2], color=color, alpha=0.6, linewidth=1)
    
    axes[0].set_ylabel('X Position (m)')
    axes[0].set_title('Position Profiles Over Time')
    axes[0].legend(loc='upper right', fontsize=6, ncol=4)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Y Position (m)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('Z Position (m)')
    axes[2].set_xlabel('Normalized Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, 'position_vs_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {section_dir}/position_vs_time.png")
    
    # 3. Velocity analysis and POI identification
    print("\n3. Analyzing velocity and identifying Points of Interest...")
    poi_data = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    demo_name = list(demos.keys())[0]  # Use first demo for detailed analysis
    times, trajectory = demos[demo_name]
    positions = trajectory[:, :3]
    
    # Calculate velocity
    dt = np.diff(times)
    dt[dt < 0.001] = 0.001
    velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
    velocity_mag = np.linalg.norm(velocities, axis=1)
    
    # Smooth velocity
    win = min(31, len(velocity_mag) - 1)
    if win % 2 == 0:
        win -= 1
    if win >= 5:
        velocity_smooth = savgol_filter(velocity_mag, win, 3)
    else:
        velocity_smooth = velocity_mag
    
    # Identify pause points (potential pick/place events)
    pause_threshold = 0.02
    
    # Find Z-minima for pick/place detection
    z = positions[:, 2]
    z_minima = []
    window = 20
    for i in range(window, len(z) - window):
        if z[i] < z[i-window:i].min() and z[i] < z[i+1:i+window+1].min():
            if velocity_smooth[min(i, len(velocity_smooth)-1)] < 0.05:
                z_minima.append(i)
    
    # Store POI data for this demo
    poi_data[demo_name] = {
        'z_minima_indices': z_minima,
        'z_minima_positions': positions[z_minima] if z_minima else np.array([]),
        'z_minima_times': times[z_minima] if z_minima else np.array([])
    }
    
    # Collect POI for ALL demonstrations
    for name, (demo_times, demo_trajectory) in demos.items():
        if name == demo_name:
            continue  # Already processed
        
        demo_positions = demo_trajectory[:, :3]
        demo_z = demo_positions[:, 2]
        
        # Calculate velocity for this demo
        demo_dt = np.diff(demo_times)
        demo_dt[demo_dt < 0.001] = 0.001
        demo_velocities = np.diff(demo_positions, axis=0) / demo_dt[:, np.newaxis]
        demo_vel_mag = np.linalg.norm(demo_velocities, axis=1)
        
        # Smooth velocity
        demo_win = min(31, len(demo_vel_mag) - 1)
        if demo_win % 2 == 0:
            demo_win -= 1
        if demo_win >= 5:
            demo_vel_smooth = savgol_filter(demo_vel_mag, demo_win, 3)
        else:
            demo_vel_smooth = demo_vel_mag
        
        # Find Z-minima (pick/place events)
        demo_z_minima = []
        for i in range(window, len(demo_z) - window):
            if demo_z[i] < demo_z[i-window:i].min() and demo_z[i] < demo_z[i+1:i+window+1].min():
                if demo_vel_smooth[min(i, len(demo_vel_smooth)-1)] < 0.05:
                    demo_z_minima.append(i)
        
        poi_data[name] = {
            'z_minima_indices': demo_z_minima,
            'z_minima_positions': demo_positions[demo_z_minima] if demo_z_minima else np.array([]),
            'z_minima_times': demo_times[demo_z_minima] if demo_z_minima else np.array([])
        }
    
    # Print POI summary for all demos
    print(f"\n  POI Summary (Z-minima with low velocity):")
    total_poi = 0
    for name, data in poi_data.items():
        count = len(data['z_minima_indices'])
        total_poi += count
        print(f"    {name}: {count} POI detected")
    print(f"    Total across all demos: {total_poi}")
    
    # Plot velocity profile
    axes[0, 0].plot(times[:-1], velocity_smooth, 'b-', linewidth=1)
    axes[0, 0].axhline(y=pause_threshold, color='r', linestyle='--', label='Pause threshold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title(f'{demo_name}: Velocity Profile')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Z height with minima
    axes[0, 1].plot(times, z, 'b-', linewidth=1)
    for idx in z_minima[:6]:  # Show first 6 minima
        axes[0, 1].axvline(x=times[idx], color='r', linestyle='--', alpha=0.5)
        axes[0, 1].scatter(times[idx], z[idx], c='red', s=100, zorder=5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Z Position (m)')
    axes[0, 1].set_title(f'{demo_name}: Z-Height with POI (red = pick/place events)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3D with POI marked
    ax3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1, alpha=0.7)
    for idx in z_minima[:6]:
        ax3d.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], 
                    c='red', s=100, marker='*', zorder=5)
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title(f'{demo_name}: Trajectory with POI Markers')
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    DEMONSTRATION SUMMARY
    {'='*40}
    
    Total demonstrations: {len(demos)}
    
    {demo_name} Analysis:
    - Duration: {times[-1]:.2f} seconds
    - Points: {len(positions)}
    - X range: [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}] m
    - Y range: [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}] m  
    - Z range: [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}] m
    - Identified POI (pick/place): {len(z_minima)}
    
    Points of Interest (POI):
    - Detected by Z-minima + low velocity
    - Red markers show pick/place locations
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, 'poi_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {section_dir}/poi_analysis.png")
    
    # 4. Individual demo velocity plots
    print("\n4. Creating individual velocity profiles...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(demos.items()):
        if idx >= 11:
            break
        times, trajectory = data
        positions = trajectory[:, :3]
        
        dt = np.diff(times)
        dt[dt < 0.001] = 0.001
        velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
        vel_mag = np.linalg.norm(velocities, axis=1)
        
        axes[idx].plot(times[:-1], vel_mag, 'b-', linewidth=0.8)
        axes[idx].set_title(name, fontsize=9)
        axes[idx].set_xlabel('Time (s)', fontsize=8)
        axes[idx].set_ylabel('Vel (m/s)', fontsize=8)
        axes[idx].tick_params(labelsize=7)
        axes[idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(demos), 12):
        axes[idx].axis('off')
    
    plt.suptitle('Velocity Profiles for All Demonstrations', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, 'all_velocity_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {section_dir}/all_velocity_profiles.png")
    
    print("\n  Section A Complete!")
    return {'poi_count': len(z_minima), 'num_demos': len(demos)}


# ============================================================================
# SECTION B: DYNAMIC MOVEMENT PRIMITIVES
# ============================================================================

def identify_pick_place_segments(trajectory, times):
    """Extract pick-and-place segments from trajectory."""
    positions = trajectory[:, :3]
    time_diffs = np.diff(times)
    
    # Find significant pauses
    pause_threshold = 0.5
    position_tolerance = 0.015
    
    pauses = []
    for i in range(len(time_diffs)):
        if time_diffs[i] > pause_threshold:
            if i > 0 and i < len(positions) - 1:
                pos_change = np.linalg.norm(positions[i+1] - positions[i])
                if pos_change < position_tolerance:
                    pauses.append({'index': i, 'time': times[i], 'position': positions[i]})
    
    segments = []
    min_distance = 0.15
    min_duration = 3.0
    
    for i in range(len(pauses) - 1):
        start_idx = pauses[i]['index']
        end_idx = pauses[i + 1]['index']
        
        segment_traj = trajectory[start_idx:end_idx+1]
        segment_times = times[start_idx:end_idx+1]
        segment_times = segment_times - segment_times[0]
        
        start_pos = positions[start_idx]
        end_pos = positions[end_idx]
        distance = np.linalg.norm(end_pos - start_pos)
        duration = segment_times[-1]
        z_range = np.max(positions[start_idx:end_idx+1, 2]) - np.min(positions[start_idx:end_idx+1, 2])
        
        if distance > min_distance and duration > min_duration and z_range > 0.03:
            segments.append({
                'trajectory': segment_traj,
                'times': segment_times,
                'distance': distance,
                'duration': duration
            })
    
    return segments


def train_dmp_on_segment(segment, n_bfs=50):
    """Train DMP on a trajectory segment."""
    from pydmps import DMPs_discrete
    
    trajectory = segment['trajectory'].copy()
    times = segment['times'].copy()
    
    # Remove large gaps
    time_diffs = np.diff(times)
    gap_threshold = 0.15
    
    cleaned_times = np.zeros_like(times)
    cleaned_times[0] = 0
    nominal_dt = 0.05
    
    for i in range(len(time_diffs)):
        if time_diffs[i] > gap_threshold:
            cleaned_times[i + 1] = cleaned_times[i] + nominal_dt
        else:
            cleaned_times[i + 1] = cleaned_times[i] + time_diffs[i]
    
    times = cleaned_times
    tau = times[-1]
    
    # Resample to uniform timesteps
    dt = 0.01
    n_timesteps = max(int(tau / dt), 100)
    uniform_times = np.linspace(0, tau, n_timesteps)
    
    uniform_positions = np.zeros((n_timesteps, 3))
    for i in range(3):
        interp_func = interp1d(times, trajectory[:, i], kind='linear',
                              bounds_error=False, fill_value=(trajectory[0, i], trajectory[-1, i]))
        uniform_positions[:, i] = interp_func(uniform_times)
    
    # Train DMP
    dmp = DMPs_discrete(n_dmps=3, n_bfs=n_bfs, dt=dt)
    dmp.imitate_path(y_des=uniform_positions.T)
    
    return dmp, n_timesteps, tau


def run_section_b(demos, output_dir):
    """
    Section B: Dynamic Movement Primitives Learning and Generalization.
    
    This section:
    - Extracts pick-and-place segments from demonstrations
    - Trains DMPs on extracted segments
    - Tests generalization to new start/goal positions
    """
    print_section("SECTION B: DYNAMIC MOVEMENT PRIMITIVES (20%)")
    
    section_dir = os.path.join(output_dir, 'section_b_dmp')
    os.makedirs(section_dir, exist_ok=True)
    
    print("\n1. Extracting pick-and-place segments...")
    all_segments = []
    
    for name, (times, trajectory) in demos.items():
        if name in EXCLUDED_DEMOS:
            continue
        
        segments = identify_pick_place_segments(trajectory, times)
        for i, seg in enumerate(segments):
            seg['demo_name'] = name
            seg['segment_id'] = f"{name}_seg{i+1}"
            all_segments.append(seg)
    
    print(f"  Found {len(all_segments)} segments from {len(demos) - len(EXCLUDED_DEMOS)} demos")
    
    if len(all_segments) == 0:
        print("  WARNING: No segments found!")
        return {'segments': 0, 'dmps_trained': 0}
    
    print("\n2. Training DMPs on segments...")
    dmp_results = []
    
    fig = plt.figure(figsize=(18, 12))
    n_plots = min(len(all_segments), 8)
    
    for idx, segment in enumerate(all_segments[:8]):
        segment_id = segment['segment_id']
        print(f"  Training DMP for {segment_id}...")
        
        try:
            dmp, n_timesteps, tau = train_dmp_on_segment(segment)
            
            # Rollout
            y_track, _, _ = dmp.rollout(timesteps=n_timesteps)
            reproduced = y_track.T
            
            # Calculate error
            original = segment['trajectory'][:, :3]
            orig_t = segment['times'] / segment['times'][-1]
            repr_t = np.linspace(0, 1, n_timesteps)
            
            errors = []
            for dim in range(3):
                interp_func = interp1d(orig_t, original[:, dim], fill_value='extrapolate')
                orig_interp = interp_func(repr_t)
                errors.append((reproduced[dim] - orig_interp) ** 2)
            error = np.sqrt(np.sum(errors, axis=0))
            mean_error = np.mean(error) * 1000
            
            dmp_results.append({
                'segment_id': segment_id,
                'mean_error_mm': mean_error,
                'n_timesteps': n_timesteps
            })
            
            # Plot
            ax = fig.add_subplot(2, 4, idx + 1, projection='3d')
            ax.plot(original[:, 0], original[:, 1], original[:, 2], 'b-', linewidth=2, label='Demo')
            ax.plot(reproduced[0], reproduced[1], reproduced[2], 'r--', linewidth=1.5, label='DMP')
            ax.set_title(f'{segment_id}\nError: {mean_error:.1f}mm', fontsize=9)
            ax.legend(fontsize=7)
            
            # Save DMP parameters
            np.savez(os.path.join(section_dir, f'{segment_id}_dmp.npz'),
                    weights=dmp.w, y0=dmp.y0, goal=dmp.goal,
                    n_timesteps=n_timesteps, tau=tau)
            
            print(f"    Mean error: {mean_error:.2f}mm")
            
        except Exception as e:
            print(f"    ERROR: {e}")
    
    plt.suptitle('DMP Training Results: Original (blue) vs Reproduced (red)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, 'dmp_training_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {section_dir}/dmp_training_results.png")
    
    # 3. Test generalization
    print("\n3. Testing DMP generalization to new positions...")
    
    if len(all_segments) > 0 and len(dmp_results) > 0:
        # Use first segment for generalization demo
        segment = all_segments[0]
        dmp, n_timesteps, tau = train_dmp_on_segment(segment)
        
        original_start = segment['trajectory'][0, :3]
        original_goal = segment['trajectory'][-1, :3]
        
        variations = [
            ('Original', original_start, original_goal),
            ('X-Y Shift', original_start + [0.1, -0.15, 0], original_goal + [0.1, -0.15, 0]),
            ('Z Shift', original_start + [0, 0, 0.08], original_goal + [0, 0, 0.08]),
            ('Diagonal', original_start + [-0.12, 0.1, -0.05], original_goal + [-0.12, 0.1, -0.05])
        ]
        
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        colors = ['blue', 'green', 'orange', 'purple']
        for (name, start, goal), color in zip(variations, colors):
            dmp.y0 = start
            dmp.goal = goal
            y_track, _, _ = dmp.rollout(timesteps=n_timesteps)
            traj = y_track.T
            ax.plot(traj[0], traj[1], traj[2], color=color, linewidth=2, label=name)
            ax.scatter(start[0], start[1], start[2], c=color, s=100, marker='o')
            ax.scatter(goal[0], goal[1], goal[2], c=color, s=100, marker='*')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('DMP Generalization to New Start/Goal Positions')
        ax.legend()
        
        # Summary table
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        
        table_data = [['Variation', 'Start X', 'Start Y', 'Start Z', 'Goal X', 'Goal Y', 'Goal Z']]
        for name, start, goal in variations:
            table_data.append([name, f'{start[0]:.3f}', f'{start[1]:.3f}', f'{start[2]:.3f}',
                              f'{goal[0]:.3f}', f'{goal[1]:.3f}', f'{goal[2]:.3f}'])
        
        table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax2.set_title('Generalization Test Parameters')
        
        plt.tight_layout()
        plt.savefig(os.path.join(section_dir, 'dmp_generalization.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {section_dir}/dmp_generalization.png")
    
    avg_error = np.mean([r['mean_error_mm'] for r in dmp_results]) if dmp_results else 0
    print(f"\n  Section B Complete! Average error: {avg_error:.2f}mm")
    
    return {'segments': len(all_segments), 'dmps_trained': len(dmp_results), 'avg_error_mm': avg_error}


# ============================================================================
# SECTION C: DYNAMIC TIME WARPING
# ============================================================================

def run_section_c(demos, output_dir):
    """
    Section C: Dynamic Time Warping for Trajectory Alignment.
    
    This section:
    - Computes DTW distance between all trajectory pairs
    - Creates similarity matrix/heatmap
    - Identifies most similar demonstration pairs
    """
    print_section("SECTION C: DYNAMIC TIME WARPING ALIGNMENT (20%)")
    
    section_dir = os.path.join(output_dir, 'section_c_dtw')
    os.makedirs(section_dir, exist_ok=True)
    
    try:
        from dtw import dtw as dtw_func
        has_dtw = True
    except ImportError:
        has_dtw = False
        print("  WARNING: dtw package not installed. Using custom implementation.")
    
    print("\n1. Computing DTW distances between all trajectory pairs...")
    
    # Extract just positions
    trajectories = {}
    for name, (times, trajectory) in demos.items():
        trajectories[name] = trajectory[:, :3]
    
    demo_names = list(trajectories.keys())
    n_demos = len(demo_names)
    
    # Compute DTW for all pairs
    results = {}
    
    for i, name1 in enumerate(demo_names):
        for j, name2 in enumerate(demo_names):
            if j <= i:
                continue
            
            traj1 = trajectories[name1]
            traj2 = trajectories[name2]
            
            # Compute distance matrix
            dist_matrix = cdist(traj1, traj2, metric='euclidean')
            
            if has_dtw:
                alignment = dtw_func(dist_matrix)
                d = alignment.distance
                path_len = len(alignment.index1)
            else:
                # Simple DTW implementation
                n, m = dist_matrix.shape
                dtw_matrix = np.full((n + 1, m + 1), np.inf)
                dtw_matrix[0, 0] = 0
                
                for ii in range(1, n + 1):
                    for jj in range(1, m + 1):
                        cost = dist_matrix[ii-1, jj-1]
                        dtw_matrix[ii, jj] = cost + min(
                            dtw_matrix[ii-1, jj],
                            dtw_matrix[ii, jj-1],
                            dtw_matrix[ii-1, jj-1]
                        )
                d = dtw_matrix[n, m]
                path_len = n + m
            
            normalized = d / path_len
            results[(name1, name2)] = {'distance': d, 'normalized': normalized}
    
    # Sort by similarity
    sorted_pairs = sorted(results.keys(), key=lambda k: results[k]['normalized'])
    
    # Calculate similarity scores
    max_dist = max(r['normalized'] for r in results.values())
    min_dist = min(r['normalized'] for r in results.values())
    
    print("\n  Top 5 Most Similar Pairs:")
    for rank, pair in enumerate(sorted_pairs[:5], 1):
        data = results[pair]
        similarity = 1 - (data['normalized'] - min_dist) / (max_dist - min_dist) if max_dist != min_dist else 1
        print(f"    {rank}. {pair[0]} vs {pair[1]}: similarity = {similarity:.3f}")
    
    # Create similarity matrix
    print("\n2. Creating similarity matrix visualization...")
    similarity_matrix = np.zeros((n_demos, n_demos))
    
    for i, name1 in enumerate(demo_names):
        for j, name2 in enumerate(demo_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif (name1, name2) in results:
                norm = results[(name1, name2)]['normalized']
                similarity_matrix[i, j] = 1 - (norm - min_dist) / (max_dist - min_dist) if max_dist != min_dist else 1
                similarity_matrix[j, i] = similarity_matrix[i, j]
            elif (name2, name1) in results:
                norm = results[(name2, name1)]['normalized']
                similarity_matrix[i, j] = 1 - (norm - min_dist) / (max_dist - min_dist) if max_dist != min_dist else 1
                similarity_matrix[j, i] = similarity_matrix[i, j]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Heatmap
    im = axes[0].imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_xticks(range(n_demos))
    axes[0].set_yticks(range(n_demos))
    axes[0].set_xticklabels(demo_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(demo_names, fontsize=8)
    axes[0].set_title('Trajectory Similarity Matrix\n(1 = identical, 0 = most different)')
    plt.colorbar(im, ax=axes[0], label='Similarity')
    
    for i in range(n_demos):
        for j in range(n_demos):
            color = 'black' if similarity_matrix[i, j] > 0.5 else 'white'
            axes[0].text(j, i, f'{similarity_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    
    # 3D plot of most similar pairs
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for k, pair in enumerate(sorted_pairs[:3]):
        t1 = trajectories[pair[0]]
        t2 = trajectories[pair[1]]
        ax3d.plot(t1[:, 0], t1[:, 1], t1[:, 2], color=colors[k*2 % len(colors)], 
                 linewidth=2, label=pair[0])
        ax3d.plot(t2[:, 0], t2[:, 1], t2[:, 2], color=colors[(k*2+1) % len(colors)], 
                 linewidth=2, linestyle='--', label=pair[1])
    
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('Top 3 Most Similar Demo Pairs')
    ax3d.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, 'dtw_similarity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {section_dir}/dtw_similarity_analysis.png")
    
    # Save results
    np.savez(os.path.join(section_dir, 'dtw_results.npz'),
            similarity_matrix=similarity_matrix,
            demo_names=demo_names,
            sorted_pairs=sorted_pairs[:10])
    print(f"  Saved: {section_dir}/dtw_results.npz")
    
    print("\n  Section C Complete!")
    return {'pairs_compared': len(results), 'most_similar': sorted_pairs[0]}


# ============================================================================
# SECTION D: GMM/GMR LEARNING
# ============================================================================

def fit_gmm(data, n_components, random_state=42):
    """Fit Gaussian Mixture Model."""
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                         random_state=random_state, n_init=5, max_iter=200)
    gmm.fit(data)
    return gmm


def gmr_predict(gmm, input_data, input_dims, output_dims):
    """Gaussian Mixture Regression prediction."""
    n_samples = input_data.shape[0]
    n_output = len(output_dims)
    
    predicted_mean = np.zeros((n_samples, n_output))
    
    means = gmm.means_
    covars = gmm.covariances_
    weights = gmm.weights_
    n_components = len(weights)
    
    for i in range(n_samples):
        x_in = input_data[i]
        
        h = np.zeros(n_components)
        for k in range(n_components):
            mu_in = means[k, input_dims]
            sigma_in = covars[k][np.ix_(input_dims, input_dims)]
            
            try:
                diff = x_in - mu_in
                sigma_in_inv = np.linalg.inv(sigma_in)
                exp_term = -0.5 * diff @ sigma_in_inv @ diff
                det = np.linalg.det(sigma_in)
                h[k] = weights[k] * np.exp(exp_term) / np.sqrt((2*np.pi)**len(input_dims) * det + 1e-10)
            except:
                h[k] = 1e-10
        
        h = h / (np.sum(h) + 1e-10)
        
        mu_out = np.zeros(n_output)
        for k in range(n_components):
            mu_in_k = means[k, input_dims]
            mu_out_k = means[k, output_dims]
            sigma_in_k = covars[k][np.ix_(input_dims, input_dims)]
            sigma_oi_k = covars[k][np.ix_(output_dims, input_dims)]
            
            try:
                sigma_in_inv = np.linalg.inv(sigma_in_k)
                cond_mean = mu_out_k + sigma_oi_k @ sigma_in_inv @ (x_in - mu_in_k)
                mu_out += h[k] * cond_mean
            except:
                pass
        
        predicted_mean[i] = mu_out
    
    return predicted_mean


def run_section_d(demos, output_dir):
    """
    Section D: GMM/GMR Learning from Multiple Demonstrations.
    
    This section:
    - Aligns and combines multiple demonstrations
    - Finds optimal number of Gaussians using BIC
    - Trains GMM and generates trajectory via GMR
    - Compares piecewise vs complete trajectory learning
    """
    print_section("SECTION D: GMM/GMR LEARNING (30%)")
    
    section_dir = os.path.join(output_dir, 'section_d_gmm')
    os.makedirs(section_dir, exist_ok=True)
    
    print("\n1. Preparing aligned trajectory data...")
    
    # Resample all trajectories
    n_resample = 200
    aligned_demos = {}
    
    for name, (times, trajectory) in demos.items():
        positions = trajectory[:, :3]
        t_uniform, pos_resampled = resample_trajectory(times, positions, n_resample)
        aligned_demos[name] = (t_uniform, pos_resampled)
    
    print(f"  Aligned {len(aligned_demos)} demos to {n_resample} points each")
    
    # Stack all data for GMM training
    all_data = []
    for name, (times, positions) in aligned_demos.items():
        demo_data = np.hstack([times.reshape(-1, 1), positions])
        all_data.append(demo_data)
    
    all_data = np.vstack(all_data)
    print(f"  Total training data: {all_data.shape[0]} samples, {all_data.shape[1]} dimensions")
    
    # Find optimal number of Gaussians
    print("\n2. Finding optimal number of Gaussians (BIC)...")
    bic_scores = {}
    
    for n in range(2, 16):
        try:
            gmm = fit_gmm(all_data, n)
            bic = gmm.bic(all_data)
            bic_scores[n] = bic
            print(f"    n={n}: BIC={bic:.2f}")
        except Exception as e:
            print(f"    n={n}: Failed")
    
    optimal_n = min(bic_scores, key=bic_scores.get)
    print(f"\n  Optimal number of Gaussians: {optimal_n}")
    
    # Train final GMM
    print("\n3. Training GMM with optimal components...")
    gmm = fit_gmm(all_data, optimal_n)
    print(f"  GMM converged: {gmm.converged_}")
    
    # GMR prediction
    print("\n4. Generating trajectory using GMR...")
    t_query = np.linspace(0, 1, n_resample).reshape(-1, 1)
    gmr_trajectory = gmr_predict(gmm, t_query, input_dims=[0], output_dims=[1, 2, 3])
    
    # Compute error
    total_error = 0
    for name, (times, positions) in aligned_demos.items():
        error = np.mean(np.linalg.norm(gmr_trajectory - positions, axis=1))
        total_error += error
    avg_error = total_error / len(aligned_demos) * 1000
    print(f"\n  Average reproduction error: {avg_error:.2f}mm")
    
    # Visualization
    print("\n5. Creating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # BIC plot
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(list(bic_scores.keys()), list(bic_scores.values()), 'b-o', linewidth=2)
    ax1.axvline(optimal_n, color='r', linestyle='--', label=f'Optimal: {optimal_n}')
    ax1.set_xlabel('Number of Gaussians')
    ax1.set_ylabel('BIC Score')
    ax1.set_title('BIC Score vs Number of Gaussians')
    ax1.legend()
    ax1.grid(True)
    
    # 3D comparison
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    for name, (times, positions) in aligned_demos.items():
        ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.4, linewidth=1)
    ax2.plot(gmr_trajectory[:, 0], gmr_trajectory[:, 1], gmr_trajectory[:, 2], 
            'k-', linewidth=3, label='GMR Output')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Demonstrations vs GMR Reproduction')
    ax2.legend()
    
    # Position profiles
    t_gmr = np.linspace(0, 1, len(gmr_trajectory))
    
    ax3 = fig.add_subplot(2, 3, 3)
    for name, (times, positions) in aligned_demos.items():
        ax3.plot(times, positions[:, 0], alpha=0.4, linewidth=1)
    ax3.plot(t_gmr, gmr_trajectory[:, 0], 'k-', linewidth=2, label='GMR')
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('X Position (m)')
    ax3.set_title('X Position: Demos vs GMR')
    ax3.grid(True)
    
    ax4 = fig.add_subplot(2, 3, 4)
    for name, (times, positions) in aligned_demos.items():
        ax4.plot(times, positions[:, 1], alpha=0.4, linewidth=1)
    ax4.plot(t_gmr, gmr_trajectory[:, 1], 'k-', linewidth=2, label='GMR')
    ax4.set_xlabel('Normalized Time')
    ax4.set_ylabel('Y Position (m)')
    ax4.set_title('Y Position: Demos vs GMR')
    ax4.grid(True)
    
    ax5 = fig.add_subplot(2, 3, 5)
    for name, (times, positions) in aligned_demos.items():
        ax5.plot(times, positions[:, 2], alpha=0.4, linewidth=1)
    ax5.plot(t_gmr, gmr_trajectory[:, 2], 'k-', linewidth=2, label='GMR')
    ax5.set_xlabel('Normalized Time')
    ax5.set_ylabel('Z Position (m)')
    ax5.set_title('Z Position: Demos vs GMR')
    ax5.grid(True)
    
    # GMM components visualization
    ax6 = fig.add_subplot(2, 3, 6)
    from matplotlib.patches import Ellipse
    
    for name, (times, positions) in aligned_demos.items():
        ax6.scatter(times, positions[:, 0], alpha=0.2, s=5)
    
    for k in range(min(gmm.n_components, 15)):
        mean = gmm.means_[k]
        cov = gmm.covariances_[k]
        
        t_mean, x_mean = mean[0], mean[1]
        cov_2d = cov[np.ix_([0, 1], [0, 1])]
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 4 * np.sqrt(np.abs(eigenvalues))
            
            ellipse = Ellipse(xy=(t_mean, x_mean), width=width, height=height,
                            angle=angle, alpha=0.3, color='red')
            ax6.add_patch(ellipse)
            ax6.scatter(t_mean, x_mean, c='red', s=30, marker='x')
        except:
            pass
    
    ax6.set_xlabel('Normalized Time')
    ax6.set_ylabel('X Position (m)')
    ax6.set_title(f'GMM Components (n={optimal_n})')
    ax6.set_xlim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, 'gmm_gmr_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {section_dir}/gmm_gmr_results.png")
    
    # Save results
    np.savez(os.path.join(section_dir, 'gmm_gmr_results.npz'),
            gmr_trajectory=gmr_trajectory,
            optimal_n=optimal_n,
            bic_scores=bic_scores,
            gmm_means=gmm.means_,
            gmm_weights=gmm.weights_)
    print(f"  Saved: {section_dir}/gmm_gmr_results.npz")
    
    print("\n  Section D Complete!")
    return {'optimal_gaussians': optimal_n, 'avg_error_mm': avg_error}


# ============================================================================
# SECTION E: ADAPTATION AND OBSTACLE AVOIDANCE
# ============================================================================

class ObstacleAvoidance:
    """Potential field obstacle avoidance."""
    
    def __init__(self, obstacles, gamma=500, influence_radius=0.12):
        self.obstacles = np.array(obstacles)
        self.gamma = gamma
        self.influence_radius = influence_radius
    
    def compute_force(self, position, velocity):
        force = np.zeros(3)
        for obs in self.obstacles:
            diff = position - obs
            dist = np.linalg.norm(diff)
            
            if dist < self.influence_radius and dist > 0.001:
                direction = diff / dist
                normalized_dist = dist / self.influence_radius
                magnitude = self.gamma * (1.0 / dist**2) * (1 - normalized_dist)**2
                force += magnitude * direction
        
        return force


def run_section_e(demos, output_dir):
    """
    Section E: Adaptation with Different Pick/Place Positions and Obstacle Avoidance.
    
    This section:
    - Tests DMP adaptation to different start/goal positions
    - Implements obstacle avoidance using potential fields
    - Demonstrates trajectory modification around obstacles
    """
    print_section("SECTION E: ADAPTATION AND OBSTACLE AVOIDANCE (20%)")
    
    section_dir = os.path.join(output_dir, 'section_e_adaptation')
    os.makedirs(section_dir, exist_ok=True)
    
    from pydmps import DMPs_discrete
    
    # Get a sample trajectory for testing
    demo_name = [n for n in demos.keys() if n not in EXCLUDED_DEMOS][0]
    times, trajectory = demos[demo_name]
    positions = trajectory[:, :3]
    
    # Resample for DMP training
    dt = 0.01
    duration = times[-1] - times[0]
    n_timesteps = int(duration / dt)
    
    uniform_times = np.linspace(0, duration, n_timesteps)
    uniform_positions = np.zeros((n_timesteps, 3))
    
    for i in range(3):
        interp_func = interp1d(times, positions[:, i], kind='linear', fill_value='extrapolate')
        uniform_positions[:, i] = interp_func(uniform_times)
    
    # Train DMP
    print("\n1. Training DMP for obstacle avoidance demo...")
    dmp = DMPs_discrete(n_dmps=3, n_bfs=50, dt=dt)
    dmp.imitate_path(y_des=uniform_positions.T)
    
    start = uniform_positions[0]
    goal = uniform_positions[-1]
    print(f"  Start: [{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}]")
    print(f"  Goal: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")
    
    # Rollout without obstacles
    print("\n2. Rolling out DMP without obstacles...")
    dmp.reset_state()
    dmp.y0 = start
    dmp.goal = goal
    y_no_obs, _, _ = dmp.rollout(timesteps=n_timesteps)
    traj_no_obs = y_no_obs  # Shape is already (n_timesteps, 3)
    
    # Define obstacles along trajectory
    print("\n3. Setting up obstacles...")
    mid_idx = n_timesteps // 2
    traj_mid = uniform_positions[mid_idx]
    
    obstacles = [
        traj_mid + np.array([0.02, 0.03, 0.0]),
        traj_mid + np.array([-0.02, -0.02, 0.05]),
        uniform_positions[mid_idx//2] + np.array([0.01, 0.02, -0.02])
    ]
    
    print("  Obstacles placed at:")
    for i, obs in enumerate(obstacles):
        print(f"    {i+1}: [{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}]")
    
    # Rollout with obstacle avoidance
    print("\n4. Rolling out DMP with obstacle avoidance...")
    avoider = ObstacleAvoidance(obstacles, gamma=500, influence_radius=0.12)
    
    dmp.reset_state()
    dmp.y0 = start
    dmp.goal = goal
    
    traj_with_obs = np.zeros((n_timesteps, 3))
    
    for t in range(n_timesteps):
        traj_with_obs[t] = dmp.y.copy()
        y_next, dy_next, ddy = dmp.step()
        
        avoid_force = avoider.compute_force(dmp.y, dmp.dy)
        perturbation = avoid_force * dt * dt
        dmp.y = dmp.y + perturbation
    
    # Compute clearance improvement
    min_dist_no = min([min([np.linalg.norm(p - o) for p in traj_no_obs]) for o in obstacles])
    min_dist_with = min([min([np.linalg.norm(p - o) for p in traj_with_obs]) for o in obstacles])
    
    print(f"\n  Min clearance without avoidance: {min_dist_no*1000:.2f}mm")
    print(f"  Min clearance with avoidance: {min_dist_with*1000:.2f}mm")
    print(f"  Improvement: {(min_dist_with - min_dist_no)*1000:.2f}mm")
    
    # Visualization
    print("\n5. Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D comparison
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj_no_obs[:, 0], traj_no_obs[:, 1], traj_no_obs[:, 2], 'b-', linewidth=2, label='Without Avoidance')
    ax1.plot(traj_with_obs[:, 0], traj_with_obs[:, 1], traj_with_obs[:, 2], 'r-', linewidth=2, label='With Avoidance')
    
    for obs in obstacles:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = 0.02
        x = obs[0] + r * np.cos(u) * np.sin(v)
        y = obs[1] + r * np.sin(u) * np.sin(v)
        z = obs[2] + r * np.cos(v)
        ax1.plot_surface(x, y, z, color='orange', alpha=0.7)
    
    ax1.scatter(*start, c='green', s=200, marker='o', label='Start')
    ax1.scatter(*goal, c='red', s=200, marker='*', label='Goal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory: With vs Without Obstacle Avoidance')
    ax1.legend()
    
    # XY view
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(traj_no_obs[:, 0], traj_no_obs[:, 1], 'b-', linewidth=2, label='Without')
    ax2.plot(traj_with_obs[:, 0], traj_with_obs[:, 1], 'r-', linewidth=2, label='With Avoidance')
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), 0.05, color='orange', alpha=0.7)
        ax2.add_patch(circle)
    ax2.scatter(start[0], start[1], c='green', s=200, marker='o', zorder=5)
    ax2.scatter(goal[0], goal[1], c='red', s=200, marker='*', zorder=5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (XY)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    # Distance to nearest obstacle over time
    ax3 = fig.add_subplot(2, 2, 3)
    
    dist_no_obs = [min([np.linalg.norm(traj_no_obs[t] - o) for o in obstacles]) 
                   for t in range(len(traj_no_obs))]
    dist_with_obs = [min([np.linalg.norm(traj_with_obs[t] - o) for o in obstacles]) 
                     for t in range(len(traj_with_obs))]
    
    t_plot = np.linspace(0, 1, n_timesteps)
    ax3.plot(t_plot, np.array(dist_no_obs)*1000, 'b-', linewidth=2, label='Without')
    ax3.plot(t_plot, np.array(dist_with_obs)*1000, 'r-', linewidth=2, label='With Avoidance')
    ax3.axhline(y=50, color='orange', linestyle='--', label='Safety (50mm)')
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('Distance to Nearest Obstacle (mm)')
    ax3.set_title('Obstacle Clearance Over Time')
    ax3.legend()
    ax3.grid(True)
    
    # Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary = f"""
    OBSTACLE AVOIDANCE SUMMARY
    {'='*40}
    
    Number of obstacles: {len(obstacles)}
    Avoidance gamma: 500
    Influence radius: 0.12m
    
    RESULTS:
    - Min clearance (no avoidance): {min_dist_no*1000:.2f}mm
    - Min clearance (with avoidance): {min_dist_with*1000:.2f}mm
    - Improvement: {(min_dist_with - min_dist_no)*1000:.2f}mm
    
    Goal accuracy (with avoidance):
    - Final position error: {np.linalg.norm(traj_with_obs[-1] - goal)*1000:.2f}mm
    
    The potential field method successfully pushes
    the trajectory away from obstacles while
    maintaining smooth motion towards the goal.
    """
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(section_dir, 'obstacle_avoidance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {section_dir}/obstacle_avoidance.png")
    
    # Save results
    np.savez(os.path.join(section_dir, 'obstacle_avoidance_results.npz'),
            trajectory_no_obstacles=traj_no_obs,
            trajectory_with_obstacles=traj_with_obs,
            obstacles=np.array(obstacles),
            start=start, goal=goal)
    print(f"  Saved: {section_dir}/obstacle_avoidance_results.npz")
    
    print("\n  Section E Complete!")
    return {'clearance_improvement_mm': (min_dist_with - min_dist_no) * 1000}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete ECEN524 Robot Learning project."""
    
    print("\n" + "="*70)
    print("  ECEN524 ROBOT LEARNING PROJECT - COMPLETE PIPELINE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Demo folder: {DEMO_FOLDER}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all demonstrations
    print_section("LOADING DEMONSTRATION DATA")
    
    demo_files = sorted([f for f in os.listdir(DEMO_FOLDER) if f.endswith('.csv')])
    demos = {}
    
    for filename in demo_files:
        filepath = os.path.join(DEMO_FOLDER, filename)
        name = filename.replace('.csv', '')
        timestamps, trajectory = load_trajectory(filepath)
        times = parse_timestamps(timestamps)
        demos[name] = (times, trajectory)
        print(f"  Loaded {name}: {len(times)} points, {times[-1]:.2f}s duration")
    
    print(f"\nTotal demonstrations loaded: {len(demos)}")
    
    # Run all sections
    results = {}
    
    try:
        results['section_a'] = run_section_a(demos, OUTPUT_DIR)
    except Exception as e:
        print(f"  Section A Error: {e}")
        results['section_a'] = {'error': str(e)}
    
    try:
        results['section_b'] = run_section_b(demos, OUTPUT_DIR)
    except Exception as e:
        print(f"  Section B Error: {e}")
        results['section_b'] = {'error': str(e)}
    
    try:
        results['section_c'] = run_section_c(demos, OUTPUT_DIR)
    except Exception as e:
        print(f"  Section C Error: {e}")
        results['section_c'] = {'error': str(e)}
    
    try:
        results['section_d'] = run_section_d(demos, OUTPUT_DIR)
    except Exception as e:
        print(f"  Section D Error: {e}")
        results['section_d'] = {'error': str(e)}
    
    try:
        results['section_e'] = run_section_e(demos, OUTPUT_DIR)
    except Exception as e:
        print(f"  Section E Error: {e}")
        results['section_e'] = {'error': str(e)}
    
    # Final summary
    print_section("PROJECT SUMMARY")
    
    print(f"""
    Section A (Data Plotting & POI):     {'✓ Complete' if 'error' not in results.get('section_a', {}) else '✗ Failed'}
    Section B (DMP):                     {'✓ Complete' if 'error' not in results.get('section_b', {}) else '✗ Failed'}
    Section C (DTW Alignment):           {'✓ Complete' if 'error' not in results.get('section_c', {}) else '✗ Failed'}
    Section D (GMM/GMR):                 {'✓ Complete' if 'error' not in results.get('section_d', {}) else '✗ Failed'}
    Section E (Obstacle Avoidance):      {'✓ Complete' if 'error' not in results.get('section_e', {}) else '✗ Failed'}
    
    Output saved to: {OUTPUT_DIR}/
    """)
    
    # List output files
    print("  Output files generated:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), OUTPUT_DIR)
            print(f"    - {rel_path}")
    
    # Save overall results
    np.savez(os.path.join(OUTPUT_DIR, 'project_results.npz'),
            results=results, allow_pickle=True)
    print(f"\n  Results summary saved to: {OUTPUT_DIR}/project_results.npz")
    
    print("\n" + "="*70)
    print("  PROJECT EXECUTION COMPLETE!")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    main()
