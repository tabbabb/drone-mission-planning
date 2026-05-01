# -*- coding: utf-8 -*-
"""
Assignment 1 — Comprehensive Digital Map Demo
整合: DEM地形 + 山脉标注 + 威胁图层 + 雷达盲区
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Polygon as MPolygon

from src.terrain import generate_dem, get_mountain_labels, validate_requirements
from src.threats import RADAR_SITES, BUILDING_CLUSTER, NO_FLY_ZONE, get_radar_zones
from src.radar import compute_radar_viewshed
from src.visualization import (setup_map_style, plot_dem, add_north_arrow,
                                add_scale_bar, COLORS, hex_to_rgb)
from src.utils import m_to_km, MAP_WIDTH, MAP_HEIGHT, LOW_RES

setup_map_style()

print("Generating terrain...")
dem, X, Y, extent = generate_dem()
validation = validate_requirements(dem)
print(f"Terrain OK: max_elev={validation['max_elevation']:.0f}m, max_slope={validation['max_slope']:.1f}deg")

print("Computing radar viewsheds...")
viewshed, blind_combined, blind_terrain = compute_radar_viewshed(dem, res=LOW_RES)

print("Building comprehensive map...")
mtn_labels = get_mountain_labels()
det_mask, fire_mask, Xg, Yg = get_radar_zones()

fig, ax = plt.subplots(figsize=(18, 14))
ext_km = (0, MAP_WIDTH / 1000, 0, MAP_HEIGHT / 1000)

# =====================
# Layer 1: DEM with hillshade
# =====================
print("  Layer 1: Terrain hillshade + contours")
pixel_res = MAP_WIDTH / dem.shape[1]
dy, dx = np.gradient(dem, pixel_res)
azimuth, altitude = 315, 45
az_rad = np.radians(azimuth)
alt_rad = np.radians(altitude)
slope = np.arctan(np.sqrt(dx**2 + dy**2))
aspect = np.arctan2(-dx, dy)
shade = (np.cos(alt_rad) * np.cos(slope) +
         np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
shade = np.clip(shade, 0.35, 1.0)

dem_norm = (dem - dem.min()) / (dem.max() - dem.min())
terrain_rgba = plt.cm.terrain(dem_norm)
for i in range(3):
    terrain_rgba[:, :, i] *= shade

ax.imshow(terrain_rgba, extent=ext_km, origin='lower', aspect='auto', zorder=1)

# Contours
levels = np.arange(200, 2500, 200)
XX, YY = np.meshgrid(
    np.linspace(ext_km[0], ext_km[1], dem.shape[1]),
    np.linspace(ext_km[2], ext_km[3], dem.shape[0])
)
ax.contour(XX, YY, dem, levels=levels, colors=COLORS["contour"],
           linewidths=0.3, alpha=0.4, zorder=2)

# =====================
# Layer 2: Blind zones (terrain-caused)
# =====================
print("  Layer 2: Blind zones")
blind_rgba = np.zeros((*blind_terrain.shape, 4))
blind_rgba[blind_terrain, :] = [0.58, 0.0, 0.83, 0.3]  # Purple tint
ax.imshow(blind_rgba, extent=ext_km, origin='lower', aspect='auto', zorder=3)

# =====================
# Layer 3: Radar coverage
# =====================
print("  Layer 3: Radar systems")
for radar in RADAR_SITES:
    rx, ry = radar["position"]
    rkx, rky = rx / 1000, ry / 1000

    # Detection circle
    det_circle = plt.Circle((rkx, rky), radar["detection_range"] / 1000,
                             fill=True, facecolor=COLORS["radar_detection"],
                             alpha=COLORS["radar_detection_alpha"],
                             edgecolor=COLORS["radar_detection"],
                             linestyle='--', linewidth=1, zorder=3)
    ax.add_patch(det_circle)

    # Fire circle
    fire_circle = plt.Circle((rkx, rky), radar["fire_range"] / 1000,
                              fill=True, facecolor=COLORS["radar_fire"],
                              alpha=COLORS["radar_fire_alpha"],
                              edgecolor=COLORS["radar_fire"],
                              linestyle='-', linewidth=1.5, zorder=3)
    ax.add_patch(fire_circle)

    # Radar icon
    ax.plot(rkx, rky, 's', color=COLORS["radar_fire"],
            markersize=12, markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax.annotate(f"{radar['name']}\n({radar['id']})",
                xy=(rkx, rky), xytext=(8, 10), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                zorder=6)

# =====================
# Layer 4: Building cluster
# =====================
print("  Layer 4: Building cluster")
bc = BUILDING_CLUSTER
ellipse = mpatches.Ellipse(
    (bc["center"][0] / 1000, bc["center"][1] / 1000),
    width=bc["radius_x"] * 2 / 1000,
    height=bc["radius_y"] * 2 / 1000,
    facecolor=COLORS["building"], alpha=0.35,
    edgecolor='#333333', linewidth=1.5, linestyle='-',
    zorder=4,
)
ax.add_patch(ellipse)
bx = [b["x"] / 1000 for b in bc["buildings"]]
by = [b["y"] / 1000 for b in bc["buildings"]]
ax.scatter(bx, by, marker='^', color=COLORS["building"], s=25,
           edgecolors='black', linewidths=0.5, zorder=5)
ax.annotate(bc["name"], xy=(bc["center"][0] / 1000, bc["center"][1] / 1000),
            fontsize=9, fontweight='bold', ha='center', va='center', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS["building"], alpha=0.85),
            zorder=6)

# =====================
# Layer 5: No-fly zone
# =====================
print("  Layer 5: No-fly zone")
nfz = NO_FLY_ZONE
poly_km = [(x / 1000, y / 1000) for x, y in nfz["polygon"]]
polygon = MPolygon(poly_km, closed=True,
                    facecolor=COLORS["no_fly"], alpha=COLORS["no_fly_alpha"],
                    edgecolor='#8B0000', linewidth=2.5, linestyle='--',
                    hatch='////', zorder=4)
ax.add_patch(polygon)
cx_nfz = np.mean([p[0] for p in poly_km])
cy_nfz = np.mean([p[1] for p in poly_km])
ax.annotate(nfz["name"], xy=(cx_nfz, cy_nfz),
            fontsize=10, fontweight='bold', ha='center', va='center',
            color='#8B0000',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFACD', alpha=0.85),
            zorder=6)

# =====================
# Layer 6: Mountain labels
# =====================
print("  Layer 6: Mountain labels")
for ml in mtn_labels:
    ax.annotate(f"{ml['name']}\n~{ml['peak_elevation']}m",
                xy=(ml['label_x'] / 1000, ml['label_y'] / 1000),
                fontsize=9, fontweight='bold', ha='center', va='center',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#5C3317', alpha=0.75),
                zorder=6)

# =====================
# Decorations
# =====================
print("  Adding decorations...")
add_north_arrow(ax, x=0.94, y=0.93)
add_scale_bar(ax, x=0.12, y=0.05, length_km=10)

# Legend
legend_items = [
    mpatches.Patch(color=COLORS["radar_detection"], alpha=0.3, label='Radar Detection (10km)'),
    mpatches.Patch(color=COLORS["radar_fire"], alpha=0.4, label='Fire Zone (8km)'),
    mpatches.Patch(color=COLORS["blind_zone"], alpha=0.4, label='Terrain Blind Zone'),
    mpatches.Patch(color=COLORS["building"], alpha=0.4, label='Building Cluster (>=100m)'),
    mpatches.Patch(color=COLORS["no_fly"], alpha=0.4, label='No-Fly Zone'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=9,
          framealpha=0.9, edgecolor='gray', fancybox=True, ncol=2)

ax.set_xlabel('X (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (km)', fontsize=12, fontweight='bold')
ax.set_title('UAV Mission Planning — Digital Map\n'
             f'(60km x 50km, DEM res={LOW_RES}m, Max Elev {validation["max_elevation"]:.0f}m)',
             fontsize=15, fontweight='bold', pad=12)
ax.set_xlim(0, MAP_WIDTH / 1000)
ax.set_ylim(0, MAP_HEIGHT / 1000)
ax.set_aspect('equal')
ax.grid(True, alpha=0.12, color='#666666', linestyle='-', linewidth=0.5)

plt.tight_layout()
output = 'comprehensive_map.png'
plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {output}")
print("Assignment 1 map complete!")
