"""
雷达探测与盲区分析模块

计算500m飞行高度的三维视域（viewshed），标注地形遮挡盲区
"""
import numpy as np
from scipy.ndimage import label
from src.utils import (
    MAP_WIDTH, MAP_HEIGHT, LOW_RES,
    RADAR_DETECTION_RANGE,
    distance, FLIGHT_ALTITUDE,
)
from src.threats import RADAR_SITES


def _line_of_sight_batch(rx, ry, r_alt, gx_arr, gy_arr, tgt_alt, dem, res, row_arr, col_arr):
    """
    批量检查多点的LOS，返回可见性bool数组
    """
    n = len(gx_arr)
    visible = np.zeros(n, dtype=bool)

    for i in range(n):
        gx, gy = gx_arr[i], gy_arr[i]
        dist = np.sqrt((gx - rx)**2 + (gy - ry)**2)
        n_samples = max(3, int(dist / min(res, 200)))
        n_samples = min(n_samples, 500)

        sample_x = np.linspace(rx, gx, n_samples)
        sample_y = np.linspace(ry, gy, n_samples)

        c_idx = np.clip((sample_x / res).astype(int), 0, dem.shape[1] - 1)
        r_idx = np.clip((sample_y / res).astype(int), 0, dem.shape[0] - 1)
        terrain_h = dem[r_idx, c_idx]

        frac = np.linspace(0, 1, n_samples)
        los_h = r_alt + (tgt_alt - r_alt) * frac

        visible[i] = not np.any(terrain_h[1:-1] >= los_h[1:-1] + 1.0)

    return visible


def compute_radar_viewshed(dem, res=LOW_RES, flight_alt=FLIGHT_ALTITUDE):
    """
    计算所有雷达的综合视域和盲区

    Returns
    -------
    viewshed : dict, radar_id → visibility mask (bool 2D)
    blind_combined : np.ndarray, 所有雷达都看不见的区域
    blind_terrain_only : np.ndarray, 探测范围内仅因地形遮挡的盲区
    """
    cols = int(MAP_WIDTH // res)
    rows = int(MAP_HEIGHT // res)
    x_coords = np.arange(cols) * res + res / 2
    y_coords = np.arange(rows) * res + res / 2
    Xg, Yg = np.meshgrid(x_coords, y_coords)

    viewshed = {}

    for radar in RADAR_SITES:
        rx, ry = radar["position"]
        # 天线有效高度 = 地形高度 + 物理天线高度
        rx_col = int(rx / res)
        ry_row = int(ry / res)
        terrain_at_radar = dem[ry_row, rx_col]
        r_alt = terrain_at_radar + radar["antenna_height"]
        rid = radar["id"]

        print(f"  Computing viewshed for {rid} ({radar['name']})...")

        # 构建探测范围内的候选点掩膜
        dist_sq = (Xg - rx)**2 + (Yg - ry)**2
        in_range = dist_sq <= radar["detection_range"] ** 2
        in_range_indices = np.where(in_range)
        n_candidates = len(in_range_indices[0])
        print(f"    Candidates in range: {n_candidates}")

        visible = np.zeros((rows, cols), dtype=bool)

        # 分批处理
        batch_size = 2000
        for start in range(0, n_candidates, batch_size):
            end = min(start + batch_size, n_candidates)
            batch_rows = in_range_indices[0][start:end]
            batch_cols = in_range_indices[1][start:end]
            batch_x = x_coords[batch_cols]
            batch_y = y_coords[batch_rows]

            # 方位角过滤
            bearings = np.degrees(np.arctan2(batch_x - rx, batch_y - ry)) % 360
            az_diff = (bearings - radar["facing"] + 180) % 360 - 180
            in_beam = np.abs(az_diff) <= radar["beam_azimuth"]

            # 仰角过滤
            dists = np.sqrt((batch_x - rx)**2 + (batch_y - ry)**2)
            elevs = np.degrees(np.arctan2(flight_alt - r_alt, dists))
            in_elev = np.abs(elevs) <= radar["beam_elevation"]

            los_candidates = in_beam & in_elev
            if np.sum(los_candidates) == 0:
                continue

            b_rows = batch_rows[los_candidates]
            b_cols = batch_cols[los_candidates]
            b_x = batch_x[los_candidates]
            b_y = batch_y[los_candidates]

            los_results = _line_of_sight_batch(
                rx, ry, r_alt, b_x, b_y, flight_alt, dem, res, b_rows, b_cols
            )
            visible[b_rows[los_results], b_cols[los_results]] = True

            if (start // batch_size) % 20 == 0:
                print(f"      progress: {start}/{n_candidates}")

        viewshed[rid] = visible
        n_v = np.sum(visible)
        print(f"    {rid} visible: {n_v} cells ({n_v*res**2/1e6:.1f} km^2)")

    # 综合盲区
    all_visible = np.zeros((rows, cols), dtype=bool)
    for v in viewshed.values():
        all_visible |= v
    blind_combined = ~all_visible

    # 纯地形盲区
    in_range = np.zeros((rows, cols), dtype=bool)
    for radar in RADAR_SITES:
        in_range |= ((Xg - radar["position"][0])**2 + (Yg - radar["position"][1])**2
                     <= radar["detection_range"] ** 2)
    blind_terrain = in_range & blind_combined

    print(f"  Combined blind: {np.sum(blind_combined)} cells")
    print(f"  Terrain-caused blind: {np.sum(blind_terrain)} cells")

    return viewshed, blind_combined, blind_terrain


def label_blind_regions(blind_mask, min_area_km2=1.0, res=LOW_RES):
    """连通域标记盲区，过滤碎片"""
    structure = np.ones((3, 3))
    labeled, n_features = label(blind_mask, structure)
    min_pixels = int(min_area_km2 * 1e6 / (res**2))
    regions = []

    for i in range(1, n_features + 1):
        region_mask = (labeled == i)
        n_pixels = np.sum(region_mask)
        if n_pixels < min_pixels:
            continue
        r_idx, c_idx = np.where(region_mask)
        cy = np.mean(r_idx) * res + res / 2
        cx = np.mean(c_idx) * res + res / 2
        area_km2 = float(n_pixels) * float(res**2) / 1e6
        regions.append({
            "id": len(regions) + 1,
            "mask": region_mask,
            "area_km2": area_km2,
            "centroid": (cx, cy),
            "n_pixels": int(n_pixels),
        })
    return regions
