import os
import numpy as np
import pydmps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Demos to train on
DEMO_NAMES = ['demo3', 'demo8', 'demo11']
N_BFS = 50  # Number of basis functions (more = more complex movements)

os.makedirs('trained_dmps', exist_ok=True)

trained_dmps = {}

for demo_name in DEMO_NAMES:
    print(f"\n{'='*50}")
    print(f"Training DMP on {demo_name}")
    print('='*50)
    
    # Load the segment
    segment_path = f'dmp_segments/{demo_name}_segment1.npz'
    data = np.load(segment_path)
    positions = data['positions']
    times = data['times']
    
    # Calculate timing parameters
    duration = times[-1] - times[0]
    n_points = len(times)
    dt = 0.01  # Fixed dt for consistent rollout
    n_steps = int(duration / dt)
    
    print(f"  Segment duration: {duration:.2f}s")
    print(f"  Number of points: {n_points}")
    print(f"  Start: ({positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f})")
    print(f"  Goal:  ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f})")
    
    # Resample trajectory to uniform dt for DMP training
    t_uniform = np.linspace(0, duration, n_steps)
    t_orig = times - times[0]  # Normalize to start at 0
    positions_resampled = np.array([
        np.interp(t_uniform, t_orig, positions[:, i]) for i in range(3)
    ]).T
    
    # Create and train DMP
    # n_dmps=3 for X,Y,Z dimensions
    dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=N_BFS, dt=dt)
    
    # Imitate path - pydmps expects shape (n_dmps, n_timesteps)
    dmp.imitate_path(y_des=positions_resampled.T)
    
    # Store for later use
    trained_dmps[demo_name] = {
        'dmp': dmp,
        'positions': positions,
        'times': times,
        'duration': duration
    }
    
    # Save the DMP weights
    np.savez(f'trained_dmps/{demo_name}_dmp.npz',
             weights=dmp.w,
             y0=dmp.y0,
             goal=dmp.goal,
             n_bfs=N_BFS,
             dt=dt)
    print(f"  Saved: trained_dmps/{demo_name}_dmp.npz")
    
    # Reproduce trajectory with same number of timesteps
    timesteps = int(duration / dt)
    y_track, dy_track, ddy_track = dmp.rollout(timesteps=timesteps)
    
    # Calculate reproduction error by interpolating to same length
    t_orig = np.linspace(0, 1, len(positions))
    t_dmp = np.linspace(0, 1, len(y_track))
    
    # Interpolate DMP output to match original trajectory length
    y_interp = np.array([np.interp(t_orig, t_dmp, y_track[:, i]) for i in range(3)]).T
    error = np.mean(np.linalg.norm(positions - y_interp, axis=1))
    print(f"  Mean reproduction error: {error*1000:.2f} mm")

print(f"\n{'='*50}")
print("TRAINING COMPLETE")
print('='*50)

# Visualize all three DMPs
fig = plt.figure(figsize=(15, 10))

for i, demo_name in enumerate(DEMO_NAMES):
    data = trained_dmps[demo_name]
    positions = data['positions']
    dmp = data['dmp']
    duration = data['duration']
    
    timesteps = int(duration / 0.01)
    y_track, _, _ = dmp.rollout(timesteps=timesteps)
    
    # 3D plot
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Original')
    ax.plot(y_track[:, 0], y_track[:, 1], y_track[:, 2], 'r--', linewidth=2, label='DMP')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, marker='o')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='purple', s=100, marker='*')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{demo_name} - 3D Trajectory')
    ax.legend(fontsize=8)

# Show generalization example with demo11
demo_name = 'demo11'
dmp = trained_dmps[demo_name]['dmp']
positions = trained_dmps[demo_name]['positions']
duration = trained_dmps[demo_name]['duration']
timesteps = int(duration / 0.01)

# Store original goal and start
original_goal = dmp.goal.copy()
original_start = dmp.y0.copy()

# Original trajectory
y_orig, _, _ = dmp.rollout(timesteps=timesteps)

# Generalize to a new goal (shift 5cm in X)
dmp.goal = original_goal.copy()
dmp.goal[0] += 0.05
dmp.y0 = original_start.copy()
y_new_goal, _, _ = dmp.rollout(timesteps=timesteps)

# Reset and generalize with new start
dmp.goal = original_goal.copy()
dmp.y0 = original_start.copy()
dmp.y0[1] += 0.03  # Shift 3cm in Y
y_new_start, _, _ = dmp.rollout(timesteps=timesteps)

ax = fig.add_subplot(2, 3, 4, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Original Demo')
ax.plot(y_orig[:, 0], y_orig[:, 1], y_orig[:, 2], 'g--', linewidth=2, label='DMP (same goal)')
ax.plot(y_new_goal[:, 0], y_new_goal[:, 1], y_new_goal[:, 2], 'r--', linewidth=2, label='DMP (new goal +5cm X)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Generalization: New Goal')
ax.legend(fontsize=8)

ax = fig.add_subplot(2, 3, 5, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Original Demo')
ax.plot(y_new_start[:, 0], y_new_start[:, 1], y_new_start[:, 2], 'm--', linewidth=2, label='DMP (new start +3cm Y)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Generalization: New Start')
ax.legend(fontsize=8)

# Summary text
ax = fig.add_subplot(2, 3, 6)
ax.axis('off')
summary = f"""DMP Training Summary
{'='*30}

Demos trained: {', '.join(DEMO_NAMES)}
Basis functions: {N_BFS}

Saved files:
- trained_dmps/demo3_dmp.npz
- trained_dmps/demo8_dmp.npz
- trained_dmps/demo11_dmp.npz

Each file contains:
- weights: Learned forcing function weights
- y0: Start position
- goal: Goal position
- n_bfs: Number of basis functions
- dt: Time step

Usage:
  dmp.goal = new_goal  # Change target
  dmp.y0 = new_start   # Change start
  y, dy, ddy = dmp.rollout()
"""
ax.text(0.1, 0.9, summary, fontsize=10, family='monospace', va='top')

plt.tight_layout()
plt.savefig('trained_dmps/training_results.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved: trained_dmps/training_results.png")
plt.show()