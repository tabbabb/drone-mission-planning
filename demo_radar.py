# -*- coding: utf-8 -*-
"""
Radar viewshed & blind zone analysis demo
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.terrain import generate_dem
from src.radar import compute_radar_viewshed, label_blind_regions
from src.threats import RADAR_SITES
from src.utils import m_to_km, MAP_WIDTH, MAP_HEIGHT, LOW_RES

print("Loading DEM...")
dem, X, Y, extent = generate_dem()

print("Computing radar viewsheds (this may take a minute)...")
viewshed, blind_combined, blind_terrain = compute_radar_viewshed(dem, res=LOW_RES)

print("\nLabeling blind regions...")
blind_regions = label_blind_regions(blind_combined, min_area_km2=2.0)
print(f"Found {len(blind_regions)} significant blind regions:")
for br in blind_regions:
    print(f"  Region {br['id']}: {br['area_km2']:.1f} km^2, center=({br['centroid'][0]/1000:.1f}, {br['centroid'][1]/1000:.1f}) km")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

ext_km = (0, MAP_WIDTH/1000, 0, MAP_HEIGHT/1000)

titles = ['Radar R1 Viewshed', 'Radar R2 Viewshed', 'Combined Blind Zones']
data = [viewshed['R1'], viewshed['R2'], blind_combined]

for ax, title, dat in zip(axes, titles, data):
    # Background: terrain
    ax.imshow(dem, extent=ext_km, origin='lower', cmap='terrain', aspect='auto', alpha=0.6)

    if 'Blind' in title:
        # Blind zone overlay
        blind_overlay = np.zeros((*dat.shape, 4))
        blind_overlay[dat, :] = [0.58, 0.0, 0.83, 0.4]  # Purple
        ax.imshow(blind_overlay, extent=ext_km, origin='lower', aspect='auto')
    else:
        # Visible area overlay
        vis_overlay = np.zeros((*dat.shape, 4))
        vis_overlay[dat, :] = [0.0, 0.8, 0.4, 0.25]  # Green
        ax.imshow(vis_overlay, extent=ext_km, origin='lower', aspect='auto')

    # Radar site
    for radar in RADAR_SITES:
        rx, ry = radar['position']
        ax.plot(rx/1000, ry/1000, 's', color='red', markersize=8, markeredgecolor='black', zorder=5)
        det_circle = plt.Circle((rx/1000, ry/1000), radar['detection_range']/1000,
                                 fill=False, edgecolor='red', linestyle='--', linewidth=1)
        ax.add_patch(det_circle)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_xlim(0, MAP_WIDTH/1000)
    ax.set_ylim(0, MAP_HEIGHT/1000)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('demo_radar.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved: demo_radar.png")
