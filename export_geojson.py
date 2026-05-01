# -*- coding: utf-8 -*-
"""
Export threat and blind zone vector layers as GeoJSON
"""
import json
import numpy as np
from src.threats import RADAR_SITES, BUILDING_CLUSTER, NO_FLY_ZONE
from src.radar import compute_radar_viewshed, label_blind_regions
from src.terrain import generate_dem
from src.utils import MAP_WIDTH, MAP_HEIGHT, LOW_RES


def mask_to_polygons(mask, res, simplify=True):
    """Convert binary mask to polygon outlines (very simplified)"""
    from skimage import measure
    contours = measure.find_contours(mask.astype(float), 0.5)
    features = []
    for contour in contours:
        # Convert pixel coords to world coords
        coords = [(float(c[1] * res), float(c[0] * res)) for c in contour]
        if len(coords) < 4:
            continue
        # Simplify: keep every Nth point
        step = max(1, len(coords) // 50) if simplify else 1
        coords = coords[::step]
        if len(coords) < 4:
            continue
        features.append({"type": "Polygon", "coordinates": [coords]})
    return features


print("Loading DEM...")
dem, X, Y, extent = generate_dem()

print("Computing blind zones...")
viewshed, blind_combined, blind_terrain = compute_radar_viewshed(dem, res=LOW_RES)

# ============ Threats GeoJSON ============
threat_features = []

# Radar sites
for r in RADAR_SITES:
    threat_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [r["position"][0], r["position"][1]],
        },
        "properties": {
            "type": "radar_site",
            "id": r["id"],
            "name": r["name"],
            "detection_range_m": r["detection_range"],
            "fire_range_m": r["fire_range"],
            "antenna_height_m": r["antenna_height"],
            "beam_azimuth_deg": r["beam_azimuth"],
            "beam_elevation_deg": r["beam_elevation"],
            "facing_deg": r["facing"],
        },
    })

# Building cluster
bc = BUILDING_CLUSTER
threat_features.append({
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [bc["center"][0] - bc["radius_x"], bc["center"][1]],
            [bc["center"][0], bc["center"][1] + bc["radius_y"]],
            [bc["center"][0] + bc["radius_x"], bc["center"][1]],
            [bc["center"][0], bc["center"][1] - bc["radius_y"]],
            [bc["center"][0] - bc["radius_x"], bc["center"][1]],
        ]],
    },
    "properties": {
        "type": "building_cluster",
        "name": bc["name"],
        "relative_height_m": bc["relative_height"],
        "num_buildings": len(bc["buildings"]),
    },
})

# No-fly zone
threat_features.append({
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[x, y] for x, y in NO_FLY_ZONE["polygon"]] + [[NO_FLY_ZONE["polygon"][0][0], NO_FLY_ZONE["polygon"][0][1]]]],
    },
    "properties": {
        "type": "no_fly_zone",
        "name": NO_FLY_ZONE["name"],
    },
})

threats_geojson = {
    "type": "FeatureCollection",
    "features": threat_features,
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::9801"}},
}

with open("threats.geojson", "w", encoding="utf-8") as f:
    json.dump(threats_geojson, f, indent=2, ensure_ascii=False)
print("Saved: threats.geojson")

# ============ Blind Zones GeoJSON ============
print("Labeling blind regions...")
regions = label_blind_regions(blind_terrain, min_area_km2=1.0)

blind_features = []
for region in regions:
    # Simple polygon approximation (bounding box for now)
    r_idx, c_idx = np.where(region["mask"])
    min_c = c_idx.min() * LOW_RES
    max_c = (c_idx.max() + 1) * LOW_RES
    min_r = r_idx.min() * LOW_RES
    max_r = (r_idx.max() + 1) * LOW_RES

    blind_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [float(min_c), float(min_r)],
                [float(max_c), float(min_r)],
                [float(max_c), float(max_r)],
                [float(min_c), float(max_r)],
                [float(min_c), float(min_r)],
            ]],
        },
        "properties": {
            "type": "radar_blind_zone",
            "id": int(region["id"]),
            "area_km2": float(region["area_km2"]),
            "cause": "terrain_occlusion",
            "flight_altitude_m": 500,
        },
    })

blind_geojson = {
    "type": "FeatureCollection",
    "features": blind_features,
}

with open("blind_zones.geojson", "w", encoding="utf-8") as f:
    json.dump(blind_geojson, f, indent=2, ensure_ascii=False)
print(f"Saved: blind_zones.geojson ({len(blind_features)} regions)")

print("Export complete!")
