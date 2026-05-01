# -*- coding: utf-8 -*-
"""
Threat elements demo — verify all threat layers
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.terrain import generate_dem
from src.threats import (RADAR_SITES, BUILDING_CLUSTER, NO_FLY_ZONE,
                          get_radar_zones, get_building_mask, get_no_fly_mask)
from src.visualization import (setup_map_style, plot_dem, plot_radar_coverage,
                                plot_buildings, plot_no_fly_zone, add_north_arrow,
                                add_scale_bar, add_legend, COLORS)
from src.utils import m_to_km, MAP_WIDTH, MAP_HEIGHT

setup_map_style()
dem, X, Y, extent = generate_dem()
det_mask, fire_mask, Xg, Yg = get_radar_zones()

print("=== Threat Summary ===")
print(f"Radar sites: {len(RADAR_SITES)}")
for r in RADAR_SITES:
    print(f"  {r['id']} ({r['name']}): pos=({r['position'][0]/1000:.1f}, {r['position'][1]/1000:.1f}) km, "
          f"detection={r['detection_range']/1000:.0f}km, fire={r['fire_range']/1000:.0f}km")
print(f"Building cluster: {BUILDING_CLUSTER['name']}, center=({BUILDING_CLUSTER['center'][0]/1000:.1f}, {BUILDING_CLUSTER['center'][1]/1000:.1f}) km")
print(f"No-fly zone: {NO_FLY_ZONE['name']}, {len(NO_FLY_ZONE['polygon'])} vertices")

fig, ax = plt.subplots(figsize=(14, 11))

# DEM hillshade
plot_dem(ax, dem, (extent[0]/1000, extent[1]/1000, extent[2]/1000, extent[3]/1000))

# Radar
plot_radar_coverage(ax, RADAR_SITES, det_mask, fire_mask, Xg, Yg)

# Buildings
plot_buildings(ax)

# No-fly zone
plot_no_fly_zone(ax)

# Decorations
add_north_arrow(ax)
add_scale_bar(ax, length_km=10)

legend_items = [
    ("Detection Range", COLORS["radar_detection"], 'patch'),
    ("Fire Zone", COLORS["radar_fire"], 'patch'),
    ("Building Cluster", COLORS["building"], 'patch'),
    ("No-Fly Zone", COLORS["no_fly"], 'patch'),
]
add_legend(ax, legend_items, loc='lower left')

ax.grid(True, alpha=0.15)
ax.set_xlabel('X (km)', fontsize=11, fontweight='bold')
ax.set_ylabel('Y (km)', fontsize=11, fontweight='bold')
ax.set_title('Threat Layer Overview', fontsize=16, fontweight='bold')
ax.set_xlim(0, MAP_WIDTH/1000)
ax.set_ylim(0, MAP_HEIGHT/1000)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('demo_threats.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: demo_threats.png")
