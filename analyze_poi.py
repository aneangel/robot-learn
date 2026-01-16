import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from scipy.spatial.transform import Rotation

def load_trajectory(filepath):
    """Load trajectory data from CSV file."""
    timestamps = np.loadtxt(filepath, delimiter=',', skiprows=1, 
                            usecols=(0,), dtype=str)
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, 
                      usecols=(1, 2, 3, 4, 5, 6, 7))
    return timestamps, data

def parse_timestamps(timestamps):
    """Convert timestamp strings to relative seconds."""
    times = []
    for ts in timestamps:
        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
        times.append(dt.timestamp())
    
    times = np.array(times)
    times = times - times[0]
    return times

def quaternion_to_euler(quat):
    """Convert quaternion [x, y, z, w] to Euler angles [roll, pitch, yaw] in degrees."""
    r = Rotation.from_quat(quat)
    return r.as_euler('xyz', degrees=True)

def identify_pause_events(timestamps, trajectory, time_gap_threshold=0.15, position_tolerance=0.005, 
                          merge_window=2.0):
    """Identify gripper actuation points by detecting pauses in trajectory."""
    times = parse_timestamps(timestamps)
    time_diffs = np.diff(times)
    positions = trajectory[:, :3]
    quaternions = trajectory[:, 3:7]
    
    raw_pauses = []
    for i in range(len(time_diffs)):
        if time_diffs[i] > time_gap_threshold:
            if i > 0 and i < len(positions) - 1:
                position_change = np.linalg.norm(positions[i+1] - positions[i])
                if position_change < position_tolerance:
                    raw_pauses.append({
                        'index': i,
                        'time': times[i],
                        'position': positions[i],
                        'time_gap': time_diffs[i]
                    })
    
    if len(raw_pauses) == 0:
        return [], times
    
    clusters = []
    current_cluster = [raw_pauses[0]]
    
    for pause in raw_pauses[1:]:
        if pause['time'] - current_cluster[-1]['time'] < merge_window:
            current_cluster.append(pause)
        else:
            clusters.append(current_cluster)
            current_cluster = [pause]
    clusters.append(current_cluster)
    
    events = []
    actions = ['GRASP', 'RELEASE']
    
    for i, cluster in enumerate(clusters):
        best_pause = max(cluster, key=lambda p: p['time_gap'])
        idx = best_pause['index']
        action = actions[i % 2]
        orientation = quaternion_to_euler(quaternions[idx])
        
        events.append({
            'index': idx,
            'position': positions[idx],
            'orientation': orientation,
            'time_gap': best_pause['time_gap'],
            'action': action,
            'cluster_size': len(cluster)
        })
    
    print(f"   Raw pauses found: {len(raw_pauses)}, Merged into {len(events)} events")
    
    return events, times

def calculate_velocity(trajectory, times):
    """Calculate velocity magnitude at each point."""
    positions = trajectory[:, :3]
    time_diffs = np.diff(times)
    time_diffs[time_diffs < 0.001] = 0.001
    
    velocities = np.diff(positions, axis=0) / time_diffs[:, np.newaxis]
    velocity_mag = np.linalg.norm(velocities, axis=1)
    velocity_mag = np.concatenate([[0], velocity_mag])
    return velocity_mag

def identify_slow_regions(velocity, threshold_percentile=20):
    """Identify slow-moving regions."""
    threshold = np.percentile(velocity, threshold_percentile)
    slow_indices = np.where(velocity < threshold)[0]
    return slow_indices, threshold

def identify_waypoints(trajectory, velocity, min_distance=0.05):
    """Identify key waypoints using velocity minima."""
    waypoint_indices = []
    waypoint_indices.append(0)
    waypoint_indices.append(len(trajectory) - 1)
    
    for i in range(1, len(velocity) - 1):
        if velocity[i] < velocity[i-1] and velocity[i] < velocity[i+1]:
            if velocity[i] < np.percentile(velocity, 30):
                waypoint_indices.append(i)
    
    waypoint_indices = sorted(list(set(waypoint_indices)))
    
    filtered_waypoints = [waypoint_indices[0]]
    for idx in waypoint_indices[1:]:
        pos_current = trajectory[idx, :3]
        pos_last = trajectory[filtered_waypoints[-1], :3]
        if np.linalg.norm(pos_current - pos_last) > min_distance:
            filtered_waypoints.append(idx)
    
    return filtered_waypoints

def analyze_approach_retreat(trajectory, pause_events, window=20):
    """Analyze approach and retreat paths around pause events."""
    positions = trajectory[:, :3]
    approach_retreat = []
    
    for event in pause_events:
        idx = event['index']
        
        approach_start = max(0, idx - window)
        approach_path = positions[approach_start:idx]
        
        retreat_end = min(len(positions), idx + window + 1)
        retreat_path = positions[idx+1:retreat_end]
        
        if len(approach_path) > 1:
            approach_vec = positions[idx] - positions[approach_start]
            approach_direction = approach_vec / (np.linalg.norm(approach_vec) + 1e-6)
        else:
            approach_direction = np.array([0, 0, 0])
        
        if len(retreat_path) > 1:
            retreat_vec = positions[min(retreat_end-1, idx+window)] - positions[idx]
            retreat_direction = retreat_vec / (np.linalg.norm(retreat_vec) + 1e-6)
        else:
            retreat_direction = np.array([0, 0, 0])
        
        approach_retreat.append({
            'event_index': idx,
            'action': event['action'],
            'approach_path': approach_path,
            'retreat_path': retreat_path,
            'approach_direction': approach_direction,
            'retreat_direction': retreat_direction,
            'approach_distance': np.linalg.norm(approach_vec) if len(approach_path) > 1 else 0,
            'retreat_distance': np.linalg.norm(retreat_vec) if len(retreat_path) > 1 else 0
        })
    
    return approach_retreat

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
    
    pause_events, times = identify_pause_events(timestamps, trajectory)
    print(f"\n1. PAUSE EVENTS - Gripper Actuation ({len(pause_events)} found):")
    for event in pause_events:
        print(f"   - {event['action']} at index {event['index']}: "
              f"pos=({event['position'][0]:.3f}, {event['position'][1]:.3f}, {event['position'][2]:.3f}), "
              f"orient=(R:{event['orientation'][0]:.1f}°, P:{event['orientation'][1]:.1f}°, Y:{event['orientation'][2]:.1f}°), "
              f"time_gap={event['time_gap']:.3f}s")
    
    extrema = identify_position_extrema(trajectory)
    print(f"\n2. POSITION EXTREMA:")
    for key, (idx, pos) in extrema.items():
        print(f"   - {key}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) at index {idx}")
    
    print(f"\n3. VELOCITY ANALYSIS:")
    print(f"   - Max velocity: {np.max(velocity):.3f} m/s")
    print(f"   - Mean velocity: {np.mean(velocity):.3f} m/s")
    print(f"   - Min velocity: {np.min(velocity):.3f} m/s")
    slow_indices, threshold = identify_slow_regions(velocity)
    print(f"   - Slow regions (<{threshold:.3f} m/s): {len(slow_indices)} points")
    
    waypoints = identify_waypoints(trajectory, velocity)
    print(f"\n4. KEY WAYPOINTS ({len(waypoints)} identified):")
    for i, idx in enumerate(waypoints):
        pos = trajectory[idx, :3]
        vel = velocity[idx]
        print(f"   - WP{i}: index {idx}, pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), vel={vel:.3f}")
    
    approach_retreat = analyze_approach_retreat(trajectory, pause_events)
    print(f"\n5. APPROACH/RETREAT PATHS ({len(approach_retreat)} analyzed):")
    for ar in approach_retreat:
        print(f"   - {ar['action']} at index {ar['event_index']}:")
        print(f"     Approach: direction=({ar['approach_direction'][0]:.2f}, {ar['approach_direction'][1]:.2f}, {ar['approach_direction'][2]:.2f}), "
              f"distance={ar['approach_distance']:.3f}m")
        print(f"     Retreat:  direction=({ar['retreat_direction'][0]:.2f}, {ar['retreat_direction'][1]:.2f}, {ar['retreat_direction'][2]:.2f}), "
              f"distance={ar['retreat_distance']:.3f}m")
    
    return {
        'trajectory': trajectory,
        'velocity': velocity,
        'times': times,
        'pause_events': pause_events,
        'waypoints': waypoints,
        'extrema': extrema,
        'slow_indices': slow_indices,
        'approach_retreat': approach_retreat
    }

def visualize_poi(analysis_results):
    """Visualize points of interest on trajectory."""
    traj = analysis_results['trajectory']
    times = analysis_results['times']
    
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.5, linewidth=1, label='Trajectory')
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, marker='o', label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, marker='X', label='End')
    
    for i, event in enumerate(analysis_results['pause_events']):
        pos = event['position']
        color = 'purple' if event['action'] == 'GRASP' else 'orange'
        marker = '*' if event['action'] == 'GRASP' else 'v'
        label = event['action'] if i == 0 or (i > 0 and event['action'] != analysis_results['pause_events'][i-1]['action']) else ""
        ax1.scatter(pos[0], pos[1], pos[2], c=color, s=200, marker=marker, label=label)
    
    for idx in analysis_results['waypoints'][1:-1]:
        pos = traj[idx, :3]
        ax1.scatter(pos[0], pos[1], pos[2], c='cyan', s=80, marker='D', alpha=0.7)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory with Points of Interest')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, analysis_results['velocity'], 'b-', linewidth=2)
    ax2.axhline(y=np.mean(analysis_results['velocity']), color='r', linestyle='--', label='Mean velocity')
    
    for idx in analysis_results['waypoints']:
        ax2.axvline(x=times[idx], color='cyan', alpha=0.3, linestyle='-')
    
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
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1)
    ax3.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='X', label='End', zorder=5)
    
    for i, event in enumerate(analysis_results['pause_events']):
        pos = event['position']
        color = 'purple' if event['action'] == 'GRASP' else 'orange'
        marker = '*' if event['action'] == 'GRASP' else 'v'
        label = event['action'] if i == 0 or (i > 0 and event['action'] != analysis_results['pause_events'][i-1]['action']) else ""
        ax3.scatter(pos[0], pos[1], c=color, s=200, marker=marker, zorder=6, label=label)
    
    for idx in analysis_results['waypoints'][1:-1]:
        pos = traj[idx, :3]
        ax3.scatter(pos[0], pos[1], c='cyan', s=80, marker='D', alpha=0.7, zorder=4)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('XY Trajectory (Top View)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
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
    output_folder = 'analysis_results'
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("\n" + "="*60)
    print("ANALYZING ALL DEMONSTRATION TRAJECTORIES")
    print("="*60)
    
    demo_files = sorted([f for f in os.listdir(demo_folder) if f.endswith('.csv')])
    
    for filename in demo_files:
        filepath = os.path.join(demo_folder, filename)
        results = analyze_single_trajectory(filepath)
        fig = visualize_poi(results)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_analysis.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
        plt.close()
    
    print("\n" + "="*60)
    print(f"COMPLETE: Analyzed {len(demo_files)} trajectories")
    print(f"Results saved in: {output_folder}/")
    print("="*60)
