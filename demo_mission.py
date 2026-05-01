# -*- coding: utf-8 -*-
"""
Assignment 2 — UAV Path Planning Demo
4 UAVs, 4 targets, primary + backup routes
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Polygon as MPolygon

from src.terrain import generate_dem, get_mountain_labels
from src.threats import RADAR_SITES, BUILDING_CLUSTER, NO_FLY_ZONE
from src.radar import compute_radar_viewshed
from src.mission import assign_tasks, UAV_BASE, get_mission_summary
from src.planner import plan_all_routes, _path_length
from src.visualization import COLORS, add_north_arrow, add_scale_bar, setup_map_style
from src.utils import m_to_km, MAP_WIDTH, MAP_HEIGHT, LOW_RES, MIN_TURN_RADIUS

setup_map_style()

# ---- Generate terrain ----
print("Loading terrain...")
dem, X, Y, extent = generate_dem()

# ---- Radar ----
print("Computing radar...")
viewshed, blind_combined, blind_terrain = compute_radar_viewshed(dem, res=LOW_RES)

# ---- Mission assignment ----
print()
tasks = get_mission_summary()

# ---- Path planning ----
print("=" * 60)
print("Path Planning")
print("=" * 60)
routes = plan_all_routes(dem)

# ---- Visualization ----
print("\nGenerating final map...")
fig, ax = plt.subplots(figsize=(20, 16))
ext_km = (0, MAP_WIDTH / 1000, 0, MAP_HEIGHT / 1000)

# 1. DEM hillshade
pixel_res = MAP_WIDTH / dem.shape[1]
dy, dx = np.gradient(dem, pixel_res)
az_rad = np.radians(315)
alt_rad = np.radians(45)
slope_arr = np.arctan(np.sqrt(dx**2 + dy**2))
aspect = np.arctan2(-dx, dy)
shade = (np.cos(alt_rad) * np.cos(slope_arr) +
         np.sin(alt_rad) * np.sin(slope_arr) * np.cos(az_rad - aspect))
shade = np.clip(shade, 0.4, 1.0)

dem_norm = (dem - dem.min()) / (dem.max() - dem.min())
terrain_rgba = plt.cm.terrain(dem_norm)
for i in range(3):
    terrain_rgba[:, :, i] *= shade
ax.imshow(terrain_rgba, extent=ext_km, origin='lower', aspect='auto', zorder=1)

# 2. Blind zones
blind_rgba = np.zeros((*blind_terrain.shape, 4))
blind_rgba[blind_terrain, :] = [0.58, 0.0, 0.83, 0.2]
ax.imshow(blind_rgba, extent=ext_km, origin='lower', aspect='auto', zorder=2)

# 3. Radar coverage
for radar in RADAR_SITES:
    rx, ry = radar["position"]
    rkx, rky = rx / 1000, ry / 1000
    det_c = plt.Circle((rkx, rky), radar["detection_range"] / 1000,
                        fill=True, facecolor=COLORS["radar_detection"],
                        alpha=0.12, edgecolor=COLORS["radar_detection"],
                        linestyle='--', linewidth=0.8, zorder=2)
    ax.add_patch(det_c)
    fire_c = plt.Circle((rkx, rky), radar["fire_range"] / 1000,
                         fill=True, facecolor=COLORS["radar_fire"],
                         alpha=0.2, edgecolor=COLORS["radar_fire"],
                         linestyle='-', linewidth=1.2, zorder=2)
    ax.add_patch(fire_c)
    ax.plot(rkx, rky, 's', color=COLORS["radar_fire"], markersize=10,
            markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.annotate(f"{radar['name']}", xy=(rkx, rky), xytext=(7, 7),
                textcoords='offset points', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85),
                zorder=6)

# 4. Buildings
bc = BUILDING_CLUSTER
ellipse = mpatches.Ellipse(
    (bc["center"][0] / 1000, bc["center"][1] / 1000),
    width=bc["radius_x"] * 2 / 1000, height=bc["radius_y"] * 2 / 1000,
    facecolor=COLORS["building"], alpha=0.3, edgecolor='#333', linewidth=1.2, zorder=3)
ax.add_patch(ellipse)
bx = [b["x"] / 1000 for b in bc["buildings"]]
by = [b["y"] / 1000 for b in bc["buildings"]]
ax.scatter(bx, by, marker='^', color=COLORS["building"], s=18,
           edgecolors='black', linewidths=0.5, zorder=4)

# 5. No-fly zone
nfz = NO_FLY_ZONE
poly_km = [(x / 1000, y / 1000) for x, y in nfz["polygon"]]
ax.add_patch(MPolygon(poly_km, closed=True, facecolor=COLORS["no_fly"],
                       alpha=0.25, edgecolor='#8B0000', linewidth=2,
                       linestyle='--', hatch='////', zorder=3))
cx_n = np.mean([p[0] for p in poly_km])
cy_n = np.mean([p[1] for p in poly_km])
ax.annotate(nfz["name"], xy=(cx_n, cy_n), fontsize=9, fontweight='bold',
            ha='center', va='center', color='#8B0000',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFACD', alpha=0.8), zorder=5)

# 6. Mountain labels
for ml in get_mountain_labels():
    ax.annotate(ml['name'], xy=(ml['label_x'] / 1000, ml['label_y'] / 1000),
                fontsize=8, fontweight='bold', ha='center', va='center', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#5C3317', alpha=0.7), zorder=5)

# 7. Routes
route_colors = ['#00CED1', '#FF8C00', '#00FF7F', '#FF6347']
route_labels = []
for task in tasks:
    uid = task["uav_id"]
    tid = task["target_id"]
    r = routes[uid]

    # Primary
    if r["primary"]:
        px = [p[0] / 1000 for p in r["primary"]]
        py = [p[1] / 1000 for p in r["primary"]]
        ax.plot(px, py, '-', color=route_colors[uid - 1], lw=2.8,
                label=f'UAV{uid}→{tid} Primary', alpha=0.85, zorder=7)
        # Arrow at midpoint
        mid = len(px) // 2
        if mid + 1 < len(px):
            ax.annotate('', xy=(px[mid + 1], py[mid + 1]),
                        xytext=(px[mid], py[mid]),
                        arrowprops=dict(arrowstyle='->', color=route_colors[uid - 1], lw=2),
                        zorder=8)

    # Backup
    if r["backup"]:
        bx = [p[0] / 1000 for p in r["backup"]]
        by = [p[1] / 1000 for p in r["backup"]]
        ax.plot(bx, by, '--', color=route_colors[uid - 1], lw=1.8,
                label=f'UAV{uid}→{tid} Backup', alpha=0.6, zorder=6)

# 8. UAV base
ax.plot(UAV_BASE[0] / 1000, UAV_BASE[1] / 1000, '^',
        color=COLORS["start_point"], markersize=15, markeredgecolor='black',
        markeredgewidth=2, zorder=9)
ax.annotate('UAV BASE', xy=(UAV_BASE[0] / 1000, UAV_BASE[1] / 1000),
            xytext=(8, -15), textcoords='offset points',
            fontsize=9, fontweight='bold', color='darkgreen', zorder=9)

# 9. Targets
for task in tasks:
    tgt = task["target"]
    ip = task["intercept_pos"]
    if tgt["type"] == "fixed":
        marker, color, size = 's', COLORS["target_fixed"], 14
    else:
        marker, color, size = 'D', COLORS["target_moving"], 14
        # 移动目标的当前位置
        ax.plot(tgt["pos"][0] / 1000, tgt["pos"][1] / 1000, 'D',
                color='gray', markersize=10, alpha=0.4, zorder=4)
        # 移动方向箭头
        heading_rad = np.radians(tgt["heading"])
        arrow_len = 3  # km
        ax.arrow(tgt["pos"][0] / 1000, tgt["pos"][1] / 1000,
                 arrow_len * np.sin(heading_rad), arrow_len * np.cos(heading_rad),
                 head_width=1.5, head_length=2, fc='gray', ec='gray', alpha=0.5, zorder=4)

    ax.plot(ip[0] / 1000, ip[1] / 1000, marker=marker, color=color,
            markersize=size, markeredgecolor='black', markeredgewidth=1.5, zorder=9)
    ax.annotate(tgt["id"], xy=(ip[0] / 1000, ip[1] / 1000),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, fontweight='bold', color='darkred', zorder=9)

# 10. Decorations
add_north_arrow(ax, x=0.95, y=0.94)
add_scale_bar(ax, x=0.12, y=0.04, length_km=10)

# Legend
legend_items = [
    mpatches.Patch(color=COLORS["radar_fire"], alpha=0.3, label='Radar Fire Zone'),
    mpatches.Patch(color=COLORS["blind_zone"], alpha=0.3, label='Radar Blind Zone'),
    mpatches.Patch(color=COLORS["building"], alpha=0.3, label='Building Cluster'),
    mpatches.Patch(color=COLORS["no_fly"], alpha=0.3, label='No-Fly Zone'),
]
for i, task in enumerate(tasks):
    legend_items.append(plt.Line2D([0], [0], color=route_colors[i], lw=2.5,
                                    label=f'UAV{task["uav_id"]}→{task["target_id"]}'))
legend_items.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["start_point"],
                                markersize=10, label='UAV Base'))
legend_items.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS["target_fixed"],
                                markersize=10, label='Fixed Target'))
legend_items.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS["target_moving"],
                                markersize=10, label='Moving Target (Intercept)'))

ax.legend(handles=legend_items, loc='lower left', fontsize=8,
          framealpha=0.9, edgecolor='gray', fancybox=True, ncol=2)

ax.set_xlabel('X (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (km)', fontsize=12, fontweight='bold')
ax.set_title('UAV Mission Planning — 4 UAVs × 4 Targets\n'
             f'(Solid=Primary Route, Dashed=Backup, Min Turn Radius={MIN_TURN_RADIUS}m)',
             fontsize=15, fontweight='bold', pad=12)
ax.set_xlim(0, MAP_WIDTH / 1000)
ax.set_ylim(0, MAP_HEIGHT / 1000)
ax.set_aspect('equal')
ax.grid(True, alpha=0.1, color='#666', linewidth=0.5)

plt.tight_layout()
output = 'route_plan.png'
plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nSaved: {output}")
print("Assignment 2 complete!")

# ---- Route statistics ----
print("\n" + "=" * 60)
print("Route Statistics")
print("=" * 60)
for task in tasks:
    uid = task["uav_id"]
    r = routes[uid]
    p_len = _path_length(r["primary"]) / 1000 if r["primary"] else 0
    b_len = _path_length(r["backup"]) / 1000 if r["backup"] else 0
    straight = _path_length([UAV_BASE, task["intercept_pos"]]) / 1000
    print(f"UAV{uid}→{task['target_id']}: Primary={p_len:.1f}km, Backup={b_len:.1f}km, "
          f"Straight={straight:.1f}km, Ratio={p_len/straight:.2f}x" if p_len else "FAILED")
