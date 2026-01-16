import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime

def load_trajectory(filepath):
    """Load trajectory data from CSV file with timestamps."""
    # Load timestamps as strings first
    timestamps = np.loadtxt(filepath, delimiter=',', skiprows=1, 
                            usecols=(0,), dtype=str)
    # Load numeric data: x, y, z, qx, qy, qz, qw
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, 
                      usecols=(1, 2, 3, 4, 5, 6, 7))
    return timestamps, data

def parse_timestamps(timestamps):
    """Convert timestamp strings to seconds from start."""
    times = []
    for ts in timestamps:
        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
        times.append(dt.timestamp())
    
    # Convert to relative time from start
    times = np.array(times)
    times = times - times[0]
    return times

def identify_pause_events(timestamps, trajectory, time_gap_threshold=0.15, position_tolerance=0.005):
    """Identify gripper actuation points by detecting pauses in trajectory.
    
    Pauses are identified when:
    - Time gap between samples is larger than expected
    - Robot is relatively stationary (low position change)
    """
    times = parse_timestamps(timestamps)
    time_diffs = np.diff(times)
    positions = trajectory[:, :3]
    
    # Expected sample time (median of time differences)
    expected_dt = np.median(time_diffs)
    
    events = []
    
    for i in range(len(time_diffs)):
        # Check if time gap is larger than threshold
        if time_diffs[i] > time_gap_threshold:
            # Check if robot was relatively stationary
            if i > 0 and i < len(positions) - 1:
                position_change = np.linalg.norm(positions[i+1] - positions[i])
                
                if position_change < position_tolerance:
                    events.append({
                        'index': i,
                        'position': positions[i],
                        'time_gap': time_diffs[i],
                        'action': 'GRASP/RELEASE'
                    })
    
    return events, times

def calculate_velocity(trajectory, times):
    """Calculate velocity magnitude at each point using actual timestamps."""
    positions = trajectory[:, :3]
    time_diffs = np.diff(times)
    
    # Avoid division by zero
    time_diffs[time_diffs < 0.001] = 0.001
    
    velocities = np.diff(positions, axis=0) / time_diffs[:, np.newaxis]
    velocity_mag = np.linalg.norm(velocities, axis=1)
    # Prepend zero for first point
    velocity_mag = np.concatenate([[0], velocity_mag])
    return velocity_mag

def identify_slow_regions(velocity, threshold_percentile=20):
    """Identify regions where robot moves slowly (potential precision areas)."""
    threshold = np.percentile(velocity, threshold_percentile)
    slow_indices = np.where(velocity < threshold)[0]
    return slow_indices, threshold

def identify_waypoints(trajectory, velocity, min_distance=0.05):
    """Identify key waypoints using velocity minima and clustering."""
    # Find local minima in velocity
    waypoint_indices = []
    
    # Start and end are always waypoints
    waypoint_indices.append(0)
    waypoint_indices.append(len(trajectory) - 1)
    
    # Find velocity minima
    for i in range(1, len(velocity) - 1):
        if velocity[i] < velocity[i-1] and velocity[i] < velocity[i+1]:
            if velocity[i] < np.percentile(velocity, 30):
                waypoint_indices.append(i)
    
    # Remove duplicates and sort
    waypoint_indices = sorted(list(set(waypoint_indices)))
    
    # Filter out waypoints that are too close together
    filtered_waypoints = [waypoint_indices[0]]
    for idx in waypoint_indices[1:]:
        pos_current = trajectory[idx, :3]
        pos_last = trajectory[filtered_waypoints[-1], :3]
        if np.linalg.norm(pos_current - pos_last) > min_distance:
            filtered_waypoints.append(idx)
    
    return filtered_waypoints

def identify_position_extrema(trajectory):
    """Find maximum and minimum positions in each dimension."""
    positions = trajectory[:, :3]
    extrema = {
        'x_min': (np.argmin(positions[:, 0]), positions[np.argmin(positions[:, 0]), :3]),
        'x_max': (np.argmax(positions[:, 0]), positions[np.argmax(positions[:, 0]), :3]),
        'y_min': (np.argmin(positions[:, 1]), positions[np.argmin(positions[:, 1]), :3]),
        'y_max': (np.argmax(positions[:, 1]), positions[np.argmax(positions[:, 1]), :3]),
        'z_min': (np.argmin(positions[:, 2]), positions[np.argmin(positions[:, 2]), :3]),
        'z_max': (np.argmax(positions[:, 2]), positions[np.argmax(positions[:, 2]), :3]),
    }
    return extrema

def analyze_single_trajectory(filepath):
    """Complete analysis of a single trajectory."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print('='*60)
    
    timestamps, trajectory = load_trajectory(filepath)
    times = parse_timestamps(timestamps)
    velocity = calculate_velocity(trajectory, times)
    
    # 1. Pause events (gripper actuation)
    pause_events, times = identify_pause_events(timestamps, trajectory)
    print(f"\n1. PAUSE EVENTS - Gripper Actuation ({len(pause_events)} found):")
    for event in pause_events:
        print(f"   - {event['action']} at index {event['index']}: "
              f"pos=({event['position'][0]:.3f}, {event['position'][1]:.3f}, {event['position'][2]:.3f}), "
              f"time_gap={event['time_gap']:.3f}s")
    
    # 2. Position extrema
    extrema = identify_position_extrema(trajectory)
    print(f"\n2. POSITION EXTREMA:")
    for key, (idx, pos) in extrema.items():
        print(f"   - {key}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) at index {idx}")
    
    # 3. Velocity analysis
    print(f"\n3. VELOCITY ANALYSIS:")
    print(f"   - Max velocity: {np.max(velocity):.3f} m/s")
    print(f"   - Mean velocity: {np.mean(velocity):.3f} m/s")
    print(f"   - Min velocity: {np.min(velocity):.3f} m/s")
    slow_indices, threshold = identify_slow_regions(velocity)
    print(f"   - Slow regions (<{threshold:.3f} m/s): {len(slow_indices)} points")
    
    # 4. Key waypoints
    waypoints = identify_waypoints(trajectory, velocity)
    print(f"\n4. KEY WAYPOINTS ({len(waypoints)} identified):")
    for i, idx in enumerate(waypoints):
        pos = trajectory[idx, :3]
        vel = velocity[idx]
        print(f"   - WP{i}: index {idx}, pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), vel={vel:.3f}")
    
    return {
        'trajectory': trajectory,
        'velocity': velocity,
        'times': times,
        'pause_events': pause_events,
        'waypoints': waypoints,
        'extrema': extrema,
        'slow_indices': slow_indices
    }

def visualize_poi(analysis_results):
    """Visualize points of interest on trajectory."""
    traj = analysis_results['trajectory']
    times = analysis_results['times']
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.5, linewidth=1, label='Trajectory')
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, marker='o', label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, marker='X', label='End')
    
    # Plot pause events (gripper actuation)
    for i, event in enumerate(analysis_results['pause_events']):
        pos = event['position']
        ax1.scatter(pos[0], pos[1], pos[2], c='purple', s=200, marker='*', 
                   label='Gripper Action (pause)' if i == 0 else "")
    
    # Plot waypoints
    for idx in analysis_results['waypoints'][1:-1]:  # Skip start/end
        pos = traj[idx, :3]
        ax1.scatter(pos[0], pos[1], pos[2], c='cyan', s=80, marker='D', alpha=0.7)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory with Points of Interest')
    ax1.legend()
    ax1.grid(True)
    
    # Velocity profile
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, analysis_results['velocity'], 'b-', linewidth=2)
    ax2.axhline(y=np.mean(analysis_results['velocity']), color='r', linestyle='--', label='Mean velocity')
    
    # Mark waypoints on velocity plot
    for idx in analysis_results['waypoints']:
        ax2.axvline(x=times[idx], color='cyan', alpha=0.3, linestyle='-')
    
    # Mark pause events (gripper actuation)
    for event in analysis_results['pause_events']:
        idx = event['index']
        ax2.axvline(x=times[idx], color='purple', alpha=0.5, linestyle='--', linewidth=2)
        ax2.text(times[idx], ax2.get_ylim()[1]*0.9, f"{event['time_gap']:.2f}s", 
                rotation=90, fontsize=8, color='purple')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Profile with Pause Events')
    ax2.legend()
    ax2.grid(True)
    
    # XY trajectory
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1)
    ax3.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='X', label='End', zorder=5)
    
    for i, event in enumerate(analysis_results['pause_events']):
        pos = event['position']
        ax3.scatter(pos[0], pos[1], c='purple', s=200, marker='*', zorder=6,
                   label='Gripper Action (pause)' if i == 0 else "")
    
    for idx in analysis_results['waypoints'][1:-1]:
        pos = traj[idx, :3]
        ax3.scatter(pos[0], pos[1], c='cyan', s=80, marker='D', alpha=0.7, zorder=4)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('XY Trajectory (Top View)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Z height over time
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(times, traj[:, 2], 'b-', linewidth=2)
    
    for idx in analysis_results['waypoints']:
        ax4.axvline(x=times[idx], color='cyan', alpha=0.3, linestyle='-')
    
    for event in analysis_results['pause_events']:
        idx = event['index']
        ax4.axvline(x=times[idx], color='purple', alpha=0.5, linestyle='--', linewidth=2)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z Height (m)')
    ax4.set_title('Height Profile')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    demo_folder = 'demotrajectories'
    
    # Analyze first demo as example
    demo_file = os.path.join(demo_folder, 'demo1.csv')
    
    results = analyze_single_trajectory(demo_file)
    fig = visualize_poi(results)
    
    plt.show()
    
    # Optional: Analyze all trajectories
    print("\n" + "="*60)
    print("SUMMARY: Analyze all trajectories? (y/n)")
    print("="*60)
    # Uncomment below to analyze all
    # for filename in sorted(os.listdir(demo_folder)):
    #     if filename.endswith('.csv'):
    #         filepath = os.path.join(demo_folder, filename)
    #         analyze_single_trajectory(filepath)
