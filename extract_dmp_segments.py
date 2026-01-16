import os
import numpy as np
import matplotlib.pyplot as plt
from analyze_poi import load_trajectory, parse_timestamps, identify_pause_events, quaternion_to_euler
from mpl_toolkits.mplot3d import Axes3D

def extract_grasp_release_segments(trajectory, times, pause_events):
    """Extract trajectory segments from GRASP to RELEASE."""
    segments = []
    
    grasp_indices = [e for e in pause_events if e['action'] == 'GRASP']
    release_indices = [e for e in pause_events if e['action'] == 'RELEASE']
    
    print(f"\nFound {len(grasp_indices)} GRASP events and {len(release_indices)} RELEASE events")
    
    for grasp_event in grasp_indices:
        grasp_idx = grasp_event['index']
        
        next_release = None
        for release_event in release_indices:
            if release_event['index'] > grasp_idx:
                next_release = release_event
                break
        
        if next_release:
            release_idx = next_release['index']
            
            segment_trajectory = trajectory[grasp_idx:release_idx+1]
            segment_times = times[grasp_idx:release_idx+1]
            segment_times = segment_times - segment_times[0]
            
            segments.append({
                'trajectory': segment_trajectory,
                'times': segment_times,
                'positions': segment_trajectory[:, :3],
                'quaternions': segment_trajectory[:, 3:7],
                'start_index': grasp_idx,
                'end_index': release_idx,
                'grasp_event': grasp_event,
                'release_event': next_release,
                'duration': segment_times[-1]
            })
            
            print(f"\nSegment {len(segments)}:")
            print(f"  GRASP at index {grasp_idx}: pos={grasp_event['position']}")
            print(f"  RELEASE at index {release_idx}: pos={next_release['position']}")
            print(f"  Duration: {segment_times[-1]:.3f}s, Points: {len(segment_trajectory)}")
    
    return segments

def prepare_dmp_data(segment):
    """Prepare trajectory segment for DMP training."""
    y_des = segment['trajectory']
    t = segment['times']
    
    return y_des, t

def save_segment_for_dmp(segment, output_path):
    """Save segment for DMP learning."""
    y_des, t = prepare_dmp_data(segment)
    
    np.savez(output_path,
             trajectory=y_des,
             times=t,
             positions=segment['positions'],
             quaternions=segment['quaternions'],
             duration=segment['duration'],
             start_index=segment['start_index'],
             end_index=segment['end_index'])
    
    print(f"Saved DMP segment to: {output_path}")

def visualize_segment(segment, demo_name, segment_num):
    """Visualize a single grasp-release segment."""
    fig = plt.figure(figsize=(15, 5))
    
    positions = segment['positions']
    times = segment['times']
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Pick-Place Path')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='purple', s=200, marker='*', label='GRASP', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='orange', s=200, marker='v', label='RELEASE', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{demo_name} - Segment {segment_num}\nGRASP → RELEASE')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(times, positions[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(times, positions[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(times, positions[:, 2], 'b-', label='Z', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position Profiles')
    ax2.legend()
    ax2.grid(True)
    
    ax3 = fig.add_subplot(1, 3, 3)
    euler_angles = np.array([quaternion_to_euler(q) for q in segment['quaternions']])
    ax3.plot(times, euler_angles[:, 0], 'r-', label='Roll', linewidth=2)
    ax3.plot(times, euler_angles[:, 1], 'g-', label='Pitch', linewidth=2)
    ax3.plot(times, euler_angles[:, 2], 'b-', label='Yaw', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Orientation Profiles')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    demo_file = 'demotrajectories/demo11.csv'
    demo_name = 'demo11'
    
    print("="*60)
    print(f"EXTRACTING GRASP-RELEASE SEGMENTS FROM {demo_name}")
    print("="*60)
    
    timestamps, trajectory = load_trajectory(demo_file)
    times = parse_timestamps(timestamps)
    pause_events, times = identify_pause_events(timestamps, trajectory)
    segments = extract_grasp_release_segments(trajectory, times, pause_events)
    
    if len(segments) == 0:
        print("\nNo GRASP-RELEASE pairs found!")
    else:
        os.makedirs('dmp_segments', exist_ok=True)
        os.makedirs('dmp_visualizations', exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {len(segments)} GRASP-RELEASE SEGMENTS")
        print("="*60)
        
        for i, segment in enumerate(segments, 1):
            output_path = f'dmp_segments/{demo_name}_segment{i}.npz'
            save_segment_for_dmp(segment, output_path)
            fig = visualize_segment(segment, demo_name, i)
            viz_path = f'dmp_visualizations/{demo_name}_segment{i}.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {viz_path}")
            plt.close()
        
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"- Extracted {len(segments)} pick-place segments")
        print(f"- DMP data saved in: dmp_segments/")
        print(f"- Visualizations saved in: dmp_visualizations/")
        print("="*60)
