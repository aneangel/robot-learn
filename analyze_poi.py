import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

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

def identify_pick_place_events(timestamps, trajectory, velocity_threshold=0.03, 
                                z_window=15, min_event_separation=50):
    """
    Identify PICK and PLACE events using velocity + Z-minima detection.
    
    This is the robust, physics-based approach:
    - PICK/PLACE occur at local Z minima (robot descends to object/surface)
    - Velocity is near zero at these points (robot pauses for gripper)
    
    Parameters:
    - velocity_threshold: Maximum velocity (m/s) to consider as "stopped"
    - z_window: Number of points on each side to check for local minimum
    - min_event_separation: Minimum points between events to avoid duplicates
    """
    times = parse_timestamps(timestamps)
    positions = trajectory[:, :3]
    quaternions = trajectory[:, 3:7]
    z = positions[:, 2]
    
    # Calculate velocity magnitude
    dt = np.diff(times)
    dt[dt < 0.001] = 0.001
    velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
    velocity_mag = np.linalg.norm(velocities, axis=1)
    velocity_mag = np.concatenate([[0], velocity_mag])
    
    # Smooth velocity to reduce noise
    win = min(31, len(velocity_mag) - 1)
    if win % 2 == 0:
        win -= 1
    if win >= 5:
        velocity_smooth = savgol_filter(velocity_mag, win, 3)
    else:
        velocity_smooth = velocity_mag
    
    # Find candidate points: low velocity
    low_velocity_mask = velocity_smooth < velocity_threshold
    
    # Find local Z minima within a window
    z_minima = []
    for i in range(z_window, len(z) - z_window):
        window_before = z[i - z_window:i]
        window_after = z[i + 1:i + z_window + 1]
        
        # Check if this point is lower than surrounding points
        if z[i] < np.min(window_before) and z[i] < np.min(window_after):
            z_minima.append(i)
    
    # Combine criteria: Z minimum AND low velocity
    candidates = []
    for idx in z_minima:
        # Check if velocity is low in a small neighborhood
        start = max(0, idx - 5)
        end = min(len(velocity_smooth), idx + 6)
        if np.min(velocity_smooth[start:end]) < velocity_threshold:
            candidates.append({
                'index': idx,
                'z': z[idx],
                'velocity': velocity_smooth[idx],
                'time': times[idx]
            })
    
    # If no candidates found with strict criteria, relax and find the two lowest Z points
    if len(candidates) < 2:
        print("   Using fallback: finding two lowest Z points with low velocity")
        # Find indices where velocity is below threshold
        low_vel_indices = np.where(velocity_smooth < velocity_threshold * 2)[0]
        if len(low_vel_indices) > 0:
            # Sort by Z value
            sorted_by_z = sorted(low_vel_indices, key=lambda i: z[i])
            # Take the two lowest, ensuring separation
            candidates = []
            for idx in sorted_by_z:
                if len(candidates) == 0:
                    candidates.append({'index': idx, 'z': z[idx], 'velocity': velocity_smooth[idx], 'time': times[idx]})
                elif abs(idx - candidates[0]['index']) > min_event_separation:
                    candidates.append({'index': idx, 'z': z[idx], 'velocity': velocity_smooth[idx], 'time': times[idx]})
                    break
    
    # Filter to ensure minimum separation between events
    filtered_candidates = []
    for cand in candidates:
        if len(filtered_candidates) == 0:
            filtered_candidates.append(cand)
        else:
            # Check distance from all existing candidates
            min_dist = min(abs(cand['index'] - fc['index']) for fc in filtered_candidates)
            if min_dist > min_event_separation:
                filtered_candidates.append(cand)
    
    # Sort by time and label as PICK (first) and PLACE (second)
    filtered_candidates.sort(key=lambda x: x['time'])
    
    events = []
    actions = ['PICK', 'PLACE']
    
    for i, cand in enumerate(filtered_candidates[:2]):  # Only take first two events
        idx = cand['index']
        action = actions[i] if i < 2 else f'EVENT_{i}'
        orientation = quaternion_to_euler(quaternions[idx])
        
        events.append({
            'index': idx,
            'position': positions[idx],
            'orientation': orientation,
            'velocity': cand['velocity'],
            'z_height': cand['z'],
            'action': action
        })
    
    print(f"   Z-minima found: {len(z_minima)}, Low-velocity candidates: {len(candidates)}, Events: {len(events)}")
    
    return events, times, velocity_smooth


def identify_pause_events(timestamps, trajectory, time_gap_threshold=0.15, position_tolerance=0.005, 
                          merge_window=2.0):
    """
    Legacy function - identifies events by time gaps in logging.
    Use identify_pick_place_events() for robust physics-based detection.
    """
    events, times, _ = identify_pick_place_events(timestamps, trajectory)
    # Convert to old format for compatibility
    for event in events:
        if event['action'] == 'PICK':
            event['action'] = 'GRASP'
        elif event['action'] == 'PLACE':
            event['action'] = 'RELEASE'
        event['time_gap'] = 0.0  # Not applicable for new method
        event['cluster_size'] = 1
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
    
    # Use the new physics-based detection
    pick_place_events, times, velocity_smooth = identify_pick_place_events(timestamps, trajectory)
    print(f"\n1. PICK/PLACE EVENTS ({len(pick_place_events)} found):")
    for event in pick_place_events:
        print(f"   - {event['action']} at index {event['index']}: "
              f"pos=({event['position'][0]:.3f}, {event['position'][1]:.3f}, {event['position'][2]:.3f}), "
              f"Z={event['z_height']:.3f}m, vel={event['velocity']:.4f}m/s")
    
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
    
    approach_retreat = analyze_approach_retreat(trajectory, pick_place_events)
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
        'velocity_smooth': velocity_smooth,
        'times': times,
        'pick_place_events': pick_place_events,
        'waypoints': waypoints,
        'extrema': extrema,
        'slow_indices': slow_indices,
        'approach_retreat': approach_retreat
    }

def visualize_poi(analysis_results):
    """Visualize points of interest on trajectory."""
    traj = analysis_results['trajectory']
    times = analysis_results['times']
    events = analysis_results['pick_place_events']
    velocity_smooth = analysis_results.get('velocity_smooth', analysis_results['velocity'])
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.5, linewidth=1, label='Trajectory')
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, marker='o', label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, marker='X', label='End')
    
    for event in events:
        pos = event['position']
        color = 'purple' if event['action'] == 'PICK' else 'orange'
        marker = '*' if event['action'] == 'PICK' else 'v'
        ax1.scatter(pos[0], pos[1], pos[2], c=color, s=200, marker=marker, label=event['action'])
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory with PICK/PLACE Events')
    ax1.legend()
    ax1.grid(True)
    
    # Velocity profile with smoothed velocity
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, analysis_results['velocity'], 'b-', alpha=0.3, linewidth=1, label='Raw velocity')
    ax2.plot(times, velocity_smooth, 'b-', linewidth=2, label='Smoothed velocity')
    ax2.axhline(y=0.03, color='r', linestyle='--', alpha=0.5, label='Detection threshold')
    
    for event in events:
        idx = event['index']
        color = 'purple' if event['action'] == 'PICK' else 'orange'
        ax2.axvline(x=times[idx], color=color, alpha=0.7, linestyle='--', linewidth=2)
        ax2.scatter(times[idx], velocity_smooth[idx], c=color, s=100, zorder=5)
        ax2.text(times[idx], ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 0.1, 
                event['action'], rotation=90, fontsize=9, color=color, va='top')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Profile - Events at Low Velocity Points')
    ax2.legend()
    ax2.grid(True)
    
    # XY trajectory (top view)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1)
    ax3.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='X', label='End', zorder=5)
    
    for event in events:
        pos = event['position']
        color = 'purple' if event['action'] == 'PICK' else 'orange'
        marker = '*' if event['action'] == 'PICK' else 'v'
        ax3.scatter(pos[0], pos[1], c=color, s=200, marker=marker, zorder=6, label=event['action'])
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('XY Trajectory (Top View)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Z height profile - key for understanding pick/place
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(times, traj[:, 2], 'b-', linewidth=2)
    
    for event in events:
        idx = event['index']
        color = 'purple' if event['action'] == 'PICK' else 'orange'
        ax4.axvline(x=times[idx], color=color, alpha=0.5, linestyle='--', linewidth=2)
        ax4.scatter(times[idx], traj[idx, 2], c=color, s=150, marker='o', zorder=5)
        ax4.text(times[idx], traj[idx, 2] - 0.01, event['action'], 
                fontsize=9, color=color, ha='center', va='top')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z Height (m)')
    ax4.set_title('Height Profile - Events at Z Minima')
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
