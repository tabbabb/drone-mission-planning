# -*- coding: utf-8 -*-
"""
Generate 5m resolution DEM for final submission
Strategy: generate at 25m, upscale to 5m, save as GeoTIFF
"""
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
import rasterio
from rasterio.transform import from_origin
from src.terrain import generate_dem, validate_requirements
from src.utils import MAP_WIDTH, MAP_HEIGHT

INTERMEDIATE_RES = 25  # m — generate at this resolution, then upscale
FINAL_RES = 5

print(f"Generating DEM at {INTERMEDIATE_RES}m resolution...")
dem_25, X, Y, extent = generate_dem(res=INTERMEDIATE_RES)

# Validation at intermediate res
val = validate_requirements(dem_25, res=INTERMEDIATE_RES)
print(f"  Max elevation: {val['max_elevation']:.0f}m")
print(f"  Max slope: {val['max_slope']:.1f}deg")
print(f"  Meets requirements: {val['meets_requirements']}")

# Upscale to 5m
scale = INTERMEDIATE_RES / FINAL_RES  # 5x
print(f"\nUpscaling {INTERMEDIATE_RES}m → {FINAL_RES}m (scale={scale}x)...")
print(f"  Input: {dem_25.shape}")
dem_5m = zoom(dem_25, scale, order=3)  # cubic interpolation

# Light smoothing to remove interpolation artifacts
print("  Smoothing...")
dem_5m = gaussian_filter(dem_5m, sigma=0.5)

print(f"  Output: {dem_5m.shape}")
print(f"  Size: {dem_5m.nbytes / 1e6:.1f} MB")

# Save as GeoTIFF
output_path = "dem_5m.tif"
print(f"\nSaving to {output_path}...")

transform = from_origin(0, MAP_HEIGHT, FINAL_RES, FINAL_RES)

with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    height=dem_5m.shape[0],
    width=dem_5m.shape[1],
    count=1,
    dtype='float32',
    crs='+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
    transform=transform,
    compress='lzw',
    tiled=True,
    blockxsize=256,
    blockysize=256,
) as dst:
    dst.write(dem_5m.astype(np.float32), 1)

print(f"DEM saved: {output_path}")

# Also save low-res version for reference
dem_100, _, _, _ = generate_dem(res=100)
with rasterio.open(
    "dem_lowres.tif", 'w',
    driver='GTiff',
    height=dem_100.shape[0],
    width=dem_100.shape[1],
    count=1,
    dtype='float32',
    crs='+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
    transform=from_origin(0, MAP_HEIGHT, 100, 100),
    compress='lzw',
) as dst:
    dst.write(dem_100.astype(np.float32), 1)

print("Low-res DEM saved: dem_lowres.tif")
print("\nDone!")
