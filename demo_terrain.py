# -*- coding: utf-8 -*-
"""
Terrain demo — DEM + Slope + 3D view
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

from src.terrain import generate_dem, get_mountain_labels, compute_slope, validate_requirements
from src.utils import m_to_km

dem, X, Y, extent = generate_dem()
slope = compute_slope(dem)
validation = validate_requirements(dem)
mtn_labels = get_mountain_labels()

print("=== Terrain Validation ===")
print(f"Max Elevation: {validation['max_elevation']:.1f} m")
print(f"Max Slope: {validation['max_slope']:.1f} deg")
print(f"Qualified area (>=500m & >=45deg): {validation['qualified_area_km2']:.2f} km^2")
print(f"Meets requirements: {validation['meets_requirements']}")

# --- 3-panel image: DEM + Slope + 3D ---
fig = plt.figure(figsize=(20, 7))

# 1. DEM
ax1 = fig.add_subplot(1, 3, 1)
im1 = ax1.imshow(dem, extent=[0, 60, 0, 50],
                 origin='lower', cmap='terrain', aspect='auto')
ax1.set_title('DEM - Digital Elevation Model', fontsize=12, fontweight='bold')
ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)')
plt.colorbar(im1, ax=ax1, label='Elevation (m)', shrink=0.82)
for ml in mtn_labels:
    ax1.annotate(ml['name'].replace('\n',' '), xy=(ml['label_x']/1000, ml['label_y']/1000),
                 fontsize=7, color='white', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

# 2. Slope
ax2 = fig.add_subplot(1, 3, 2)
im2 = ax2.imshow(slope, extent=[0, 60, 0, 50],
                 origin='lower', cmap='YlOrRd', aspect='auto', vmax=50)
ax2.set_title('Slope Map', fontsize=12, fontweight='bold')
ax2.set_xlabel('X (km)'); ax2.set_ylabel('Y (km)')
plt.colorbar(im2, ax=ax2, label='Slope (deg)', shrink=0.82)

# 3. 3D View
from mpl_toolkits.mplot3d import Axes3D  # noqa
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# Downsample for 3D performance
s = 8
X_sub = X[::s, ::s] / 1000
Y_sub = Y[::s, ::s] / 1000
dem_sub = dem[::s, ::s]
surf = ax3.plot_surface(X_sub, Y_sub, dem_sub, cmap='terrain',
                         linewidth=0, antialiased=True, alpha=0.9,
                         rstride=1, cstride=1)
ax3.set_title('3D Terrain View', fontsize=12, fontweight='bold')
ax3.set_xlabel('X (km)'); ax3.set_ylabel('Y (km)'); ax3.set_zlabel('Elev (m)')
ax3.view_init(elev=40, azim=-55)
fig.colorbar(surf, ax=ax3, label='Elevation (m)', shrink=0.6)

plt.tight_layout()
plt.savefig('demo_terrain.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: demo_terrain.png")

# --- Large standalone 3D view ---
fig2 = plt.figure(figsize=(14, 10))
ax3d = fig2.add_subplot(111, projection='3d')
s2 = 5
X_sub2 = X[::s2, ::s2] / 1000
Y_sub2 = Y[::s2, ::s2] / 1000
dem_sub2 = dem[::s2, ::s2]

# Color by elevation
norm = plt.Normalize(dem_sub2.min(), dem_sub2.max())
colors = plt.cm.terrain(norm(dem_sub2))
surf2 = ax3d.plot_surface(X_sub2, Y_sub2, dem_sub2,
                           facecolors=colors,
                           linewidth=0, antialiased=True, alpha=0.95,
                           rstride=1, cstride=1, shade=True)
ax3d.set_title('3D Terrain — 60km x 50km Mission Area', fontsize=14, fontweight='bold', pad=20)
ax3d.set_xlabel('X (km)', fontsize=11)
ax3d.set_ylabel('Y (km)', fontsize=11)
ax3d.set_zlabel('Elevation (m)', fontsize=11)
ax3d.view_init(elev=35, azim=-50)

# Add mountain labels as 3D annotations
for ml in mtn_labels:
    lx = ml['label_x'] / 1000
    ly = ml['label_y'] / 1000
    # Sample DEM at label position
    ci = int(ml['label_x'] / 100)
    ri = int(ml['label_y'] / 100)
    lz = dem[ri, ci] + 200
    ax3d.text(lx, ly, lz, ml['name'].replace('\n',' '),
              fontsize=10, fontweight='bold', color='white',
              ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#5C3317', alpha=0.8))

mip = fig2.colorbar(surf2, ax=ax3d, label='Elevation (m)', shrink=0.5, pad=0.08)
plt.tight_layout()
plt.savefig('terrain_3d.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: terrain_3d.png")
