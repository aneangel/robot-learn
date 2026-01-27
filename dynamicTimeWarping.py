import os
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Load ALL demo trajectories from CSV files
demo_folder = 'demotrajectories'
demo_files = sorted([f for f in os.listdir(demo_folder) if f.endswith('.csv')])

trajectories = {}
for filename in demo_files:
    name = filename.replace('.csv', '')
    filepath = os.path.join(demo_folder, filename)
    # Load positions (columns 1,2,3 are x,y,z)
    positions = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1, 2, 3))
    trajectories[name] = positions
    print(f"Loaded {name}: {len(positions)} points")

demo_names = list(trajectories.keys())
n_demos = len(demo_names)

print(f"\n{'='*60}")
print(f"Computing DTW for all {n_demos} demos ({n_demos*(n_demos-1)//2} pairs)")
print('='*60)

# Compute DTW between all pairs
results = {}
for i, name1 in enumerate(demo_names):
    for j, name2 in enumerate(demo_names):
        if j <= i:
            continue
        
        traj1 = trajectories[name1]
        traj2 = trajectories[name2]
        
        # Compute pairwise distance matrix
        dist_matrix = cdist(traj1, traj2, metric='euclidean')
        
        # Compute DTW
        alignment = dtw(dist_matrix)
        
        d = alignment.distance
        path_len = len(alignment.index1)
        
        # Normalize by path length
        normalized_distance = d / path_len
        
        results[(name1, name2)] = {
            'distance': d,
            'normalized': normalized_distance,
            'path_length': path_len
        }

# Sort by normalized distance (most similar first)
sorted_pairs = sorted(results.keys(), key=lambda k: results[k]['normalized'])

# Calculate similarity scores
max_dist = max(r['normalized'] for r in results.values())
min_dist = min(r['normalized'] for r in results.values())

print("\n" + "="*60)
print("SIMILARITY RANKING (most similar first)")
print("="*60)
print(f"{'Rank':<6}{'Pair':<25}{'DTW Dist':<12}{'Normalized':<12}{'Similarity':<10}")
print("-"*60)

for rank, pair in enumerate(sorted_pairs, 1):
    data = results[pair]
    # Scale similarity to 0-1 where 1 is most similar
    similarity = 1 - (data['normalized'] - min_dist) / (max_dist - min_dist) if max_dist != min_dist else 1
    print(f"{rank:<6}{pair[0]} vs {pair[1]:<15}{data['distance']:<12.2f}{data['normalized']:<12.4f}{similarity:<10.3f}")

# Find the top 5 most similar pairs
print("\n" + "="*60)
print("TOP 5 MOST SIMILAR PAIRS")
print("="*60)
top_5 = sorted_pairs[:5]
for i, pair in enumerate(top_5, 1):
    data = results[pair]
    similarity = 1 - (data['normalized'] - min_dist) / (max_dist - min_dist) if max_dist != min_dist else 1
    print(f"{i}. {pair[0]} and {pair[1]}: similarity = {similarity:.3f}")

# Create similarity matrix for heatmap
similarity_matrix = np.zeros((n_demos, n_demos))
for i, name1 in enumerate(demo_names):
    for j, name2 in enumerate(demo_names):
        if i == j:
            similarity_matrix[i, j] = 1.0  # Self-similarity
        elif (name1, name2) in results:
            norm = results[(name1, name2)]['normalized']
            similarity_matrix[i, j] = 1 - (norm - min_dist) / (max_dist - min_dist)
            similarity_matrix[j, i] = similarity_matrix[i, j]
        elif (name2, name1) in results:
            norm = results[(name2, name1)]['normalized']
            similarity_matrix[i, j] = 1 - (norm - min_dist) / (max_dist - min_dist)
            similarity_matrix[j, i] = similarity_matrix[i, j]

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Heatmap of similarity matrix
ax1 = axes[0]
im = ax1.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
ax1.set_xticks(range(n_demos))
ax1.set_yticks(range(n_demos))
ax1.set_xticklabels(demo_names, rotation=45, ha='right')
ax1.set_yticklabels(demo_names)
ax1.set_title('Trajectory Similarity Matrix\n(1 = identical, 0 = most different)')
plt.colorbar(im, ax=ax1, label='Similarity')

# Add text annotations
for i in range(n_demos):
    for j in range(n_demos):
        ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}', ha='center', va='center', 
                fontsize=7, color='black' if similarity_matrix[i, j] > 0.5 else 'white')

# 3D plot of top similar pair
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

colors = ['b', 'r', 'g', 'm', 'c', 'y']  # Cycle more colors if needed
plotted_labels = set()

for k, pair in enumerate(sorted_pairs[:5], 1):
    t1, t2 = trajectories[pair[0]], trajectories[pair[1]]
    color1 = colors[(2*k-2) % len(colors)]
    color2 = colors[(2*k-1) % len(colors)]
    l1 = f"{pair[0]} (#{k})"
    l2 = f"{pair[1]} (#{k})"
    # Only plot unique labels in legend
    if l1 not in plotted_labels:
        ax2.plot(t1[:, 0], t1[:, 1], t1[:, 2], color=color1, linewidth=2, label=l1)
        plotted_labels.add(l1)
    else:
        ax2.plot(t1[:, 0], t1[:, 1], t1[:, 2], color=color1, linewidth=2)
    if l2 not in plotted_labels:
        ax2.plot(t2[:, 0], t2[:, 1], t2[:, 2], color=color2, linewidth=2, label=l2, linestyle='--')
        plotted_labels.add(l2)
    else:
        ax2.plot(t2[:, 0], t2[:, 1], t2[:, 2], color=color2, linewidth=2, linestyle='--')

ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.set_title('Top 5 Most Similar Demo Pairs\n(Plotted as solid/dashed lines)')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('dtw_all_demos.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved: dtw_all_demos.png")
plt.show()
