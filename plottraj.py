import os
import numpy as np
import matplotlib.pyplot as plt

# Path to demo trajectories folder
demo_folder = 'demotrajectories'

# Get all trajectory files
trajectory_files = [f for f in os.listdir(demo_folder) if f.endswith('.npy') or f.endswith('.txt') or f.endswith('.csv')]

# Create figure with four subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot each trajectory
for i, filename in enumerate(trajectory_files):
    filepath = os.path.join(demo_folder, filename)
    
    # Load trajectory data
    if filename.endswith('.npy'):
        data = np.load(filepath)
        # Assuming npy files have x, y, z columns
        time = np.arange(len(data))
        x_data = data[:, 0]
        y_data = data[:, 1]
        z_data = data[:, 2]
        # Assuming columns 3-6 are qx, qy, qz, qw
        qx_data = data[:, 3] if data.shape[1] > 3 else None
        qy_data = data[:, 4] if data.shape[1] > 4 else None
        qz_data = data[:, 5] if data.shape[1] > 5 else None
        qw_data = data[:, 6] if data.shape[1] > 6 else None
    elif filename.endswith('.txt') or filename.endswith('.csv'):
        # Load x, y, z coordinates and quaternions (columns 1-7)
        full_data = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7))
        time = np.arange(len(full_data))
        x_data = full_data[:, 0]
        y_data = full_data[:, 1]
        z_data = full_data[:, 2]
        qx_data = full_data[:, 3]
        qy_data = full_data[:, 4]
        qz_data = full_data[:, 5]
        qw_data = full_data[:, 6]
    
    # Plot x vs time
    ax1.plot(time, x_data, label=filename, alpha=0.7)
    
    # Plot y vs time
    ax2.plot(time, y_data, label=filename, alpha=0.7)
    
    # Plot z vs time
    ax3.plot(time, z_data, label=filename, alpha=0.7)
    
    # Plot quaternion components vs time
    if qx_data is not None:
        ax4.plot(time, qx_data, label=f'{filename} (qx)', alpha=0.5, linestyle='-')
        ax4.plot(time, qy_data, label=f'{filename} (qy)', alpha=0.5, linestyle='--')
        ax4.plot(time, qz_data, label=f'{filename} (qz)', alpha=0.5, linestyle='-.')
        ax4.plot(time, qw_data, label=f'{filename} (qw)', alpha=0.5, linestyle=':')

ax1.set_xlabel('Time (samples)')
ax1.set_ylabel('X')
ax1.set_title('X Position vs Time')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Time (samples)')
ax2.set_ylabel('Y')
ax2.set_title('Y Position vs Time')
ax2.legend()
ax2.grid(True)

ax3.set_xlabel('Time (samples)')
ax3.set_ylabel('Z')
ax3.set_title('Z Position vs Time')
ax3.legend()
ax3.grid(True)

ax4.set_xlabel('Time (samples)')
ax4.set_ylabel('Quaternion Value')
ax4.set_title('Quaternion Components vs Time')
ax4.legend(fontsize=6, ncol=2)
ax4.grid(True)

plt.tight_layout()
plt.show()
