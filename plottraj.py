import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter

demo_folder = 'demotrajectories'
output_folder = 'velocity_plots'
os.makedirs(output_folder, exist_ok=True)

trajectory_files = sorted([f for f in os.listdir(demo_folder) if f.endswith('.csv')])

# Smoothing parameters
WINDOW_LENGTH = 31  # Must be odd, larger = more smoothing
POLY_ORDER = 3      # Polynomial order for fitting

# Store all velocity data for combined plot
all_velocities = []

for filename in trajectory_files:
    filepath = os.path.join(demo_folder, filename)
    demo_name = filename.replace('.csv', '')
    
    # Load timestamps and positions
    timestamps = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(0,), dtype=str)
    positions = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1, 2, 3))
    
    # Parse timestamps to get time in seconds
    times = []
    for ts in timestamps:
        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
        times.append(dt.timestamp())
    times = np.array(times)
    times = times - times[0]
    
    # Calculate velocities (derivative of position)
    dt = np.diff(times)
    dt[dt < 0.001] = 0.001  # Avoid division by zero
    
    vx = np.diff(positions[:, 0]) / dt
    vy = np.diff(positions[:, 1]) / dt
    vz = np.diff(positions[:, 2]) / dt
    
    # Smooth velocities using Savitzky-Golay filter
    win = min(WINDOW_LENGTH, len(vx) - 1)
    if win % 2 == 0:
        win -= 1  # Must be odd
    if win >= 5:
        vx_smooth = savgol_filter(vx, win, POLY_ORDER)
        vy_smooth = savgol_filter(vy, win, POLY_ORDER)
        vz_smooth = savgol_filter(vz, win, POLY_ORDER)
    else:
        vx_smooth, vy_smooth, vz_smooth = vx, vy, vz
    
    # Time for velocity
    t_vel = times[:-1]
    
    # Store for combined plot
    all_velocities.append({
        'name': demo_name,
        't': t_vel,
        'vx': vx_smooth,
        'vy': vy_smooth,
        'vz': vz_smooth
    })
    
    # Create figure for this demo
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(t_vel, vx_smooth, 'r-', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title(f'{demo_name} - X Velocity')
    ax1.grid(True)
    
    ax2.plot(t_vel, vy_smooth, 'g-', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title(f'{demo_name} - Y Velocity')
    ax2.grid(True)
    
    ax3.plot(t_vel, vz_smooth, 'b-', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title(f'{demo_name} - Z Velocity')
    ax3.grid(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, f'{demo_name}_velocity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()

# Create combined plot with all demos overlapped
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

for data in all_velocities:
    ax1.plot(data['t'], data['vx'], linewidth=1.2, alpha=0.7, label=data['name'])
    ax2.plot(data['t'], data['vy'], linewidth=1.2, alpha=0.7, label=data['name'])
    ax3.plot(data['t'], data['vz'], linewidth=1.2, alpha=0.7, label=data['name'])

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('All Demos - X Velocity')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(True)

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('All Demos - Y Velocity')
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(True)

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('All Demos - Z Velocity')
ax3.legend(fontsize=7, loc='upper right')
ax3.grid(True)

plt.tight_layout()
combined_path = os.path.join(output_folder, 'all_demos_velocity.png')
plt.savefig(combined_path, dpi=150, bbox_inches='tight')
print(f'Saved: {combined_path}')
plt.close()

print(f'\nAll velocity plots saved to: {output_folder}/')
