"""
雷达探测盲区分析 — 三维视域 + 地形遮挡
"""
import numpy as np
from .utils import FLIGHT_ALT, WORK_RES
from .threats import RADARS


def _los_check(rx, ry, r_alt, gx, gy, tgt_alt, dem, res):
    """雷达→目标的视线检查"""
    dist = np.sqrt((gx - rx)**2 + (gy - ry)**2)
    n = max(3, int(dist / min(res, 200)))
    n = min(n, 500)

    sx = np.linspace(rx, gx, n)
    sy = np.linspace(ry, gy, n)
    ci = np.clip((sx / res).astype(int), 0, dem.shape[1] - 1)
    ri = np.clip((sy / res).astype(int), 0, dem.shape[0] - 1)
    terrain = dem[ri, ci]

    frac = np.linspace(0, 1, n)
    los = r_alt + (tgt_alt - r_alt) * frac

    return not np.any(terrain[1:-1] >= los[1:-1] + 1.0)


def compute_viewshed(dem, radars=None, flight_alt=FLIGHT_ALT, res=WORK_RES):
    """
    计算雷达综合视域

    Returns
    -------
    viewsheds : dict  radar_id → visibility mask (bool 2D)
    blind_all : np.ndarray  所有雷达都看不见的区域
    blind_terrain : np.ndarray  探测范围内仅因地形的盲区
    """
    if radars is None:
        radars = RADARS

    rows, cols = dem.shape
    cx = np.arange(cols) * res + res / 2
    cy = np.arange(rows) * res + res / 2
    Xg, Yg = np.meshgrid(cx, cy)

    viewsheds = {}

    for r in radars:
        rx, ry = r["位置"]
        # 有效天线高度 = 地形 + 物理天线
        rc = int(rx / res)
        rr = int(ry / res)
        r_alt = dem[rr, rc] + r["天线高度_m"]

        print(f"  计算 {r['id']} ({r['型号']}) 视域... "
              f"有效高度={r_alt:.0f}m")

        visible = np.zeros((rows, cols), dtype=bool)
        det_sq = r["探测距离_m"] ** 2

        # 探测范围内的所有格点
        in_range = np.where((Xg - rx)**2 + (Yg - ry)**2 <= det_sq)

        # 方位角过滤
        bearings = np.degrees(np.arctan2(Xg[in_range] - rx, Yg[in_range] - ry)) % 360
        az_diff = (bearings - r["主瓣朝向_deg"] + 180) % 360 - 180
        in_beam = np.abs(az_diff) <= r["波束宽度_deg"]

        # 仰角过滤
        dists = np.sqrt((Xg[in_range] - rx)**2 + (Yg[in_range] - ry)**2)
        elevs = np.degrees(np.arctan2(flight_alt - r_alt, dists))
        in_elev = np.abs(elevs) <= r["波束宽度_deg"]

        candidates = in_beam & in_elev
        print(f"    候选格点: {np.sum(candidates)} / {len(in_range[0])}")

        # LOS检查
        cand_rows = in_range[0][candidates]
        cand_cols = in_range[1][candidates]
        batch = 1000
        for start in range(0, len(cand_rows), batch):
            end = min(start + batch, len(cand_rows))
            for i in range(start, end):
                r_idx, c_idx = cand_rows[i], cand_cols[i]
                gx, gy = cx[c_idx], cy[r_idx]
                if _los_check(rx, ry, r_alt, gx, gy, flight_alt, dem, res):
                    visible[r_idx, c_idx] = True
            if start % 5000 == 0 and start > 0:
                print(f"      LOS进度: {start}/{len(cand_rows)}")

        viewsheds[r["id"]] = visible
        n_v = np.sum(visible)
        print(f"    可见: {n_v} 格点 ({float(n_v) * res**2 / 1e6:.1f} km^2)")

    # 综合盲区
    all_visible = np.zeros((rows, cols), dtype=bool)
    for v in viewsheds.values():
        all_visible |= v

    blind_all = ~all_visible

    # 分类盲区
    # 1) 探测范围内 (any radar)
    in_range = np.zeros((rows, cols), dtype=bool)
    for r in radars:
        in_range |= ((Xg - r["位置"][0])**2 + (Yg - r["位置"][1])**2
                     <= r["探测距离_m"] ** 2)

    # 2) 波束角度内 (至少一部雷达的方位+仰角覆盖)
    in_beam = np.zeros((rows, cols), dtype=bool)
    for r in radars:
        rx, ry = r["位置"]
        rc = int(rx / res); rr = int(ry / res)
        r_alt = dem[rr, rc] + r["天线高度_m"]
        d_sq = (Xg - rx)**2 + (Yg - ry)**2
        det_sq = r["探测距离_m"] ** 2
        in_det = d_sq <= det_sq
        if not np.any(in_det):
            continue
        bearings = np.degrees(np.arctan2(Xg - rx, Yg - ry)) % 360
        az_diff = (bearings - r["主瓣朝向_deg"] + 180) % 360 - 180
        in_az = np.abs(az_diff) <= r["波束宽度_deg"]
        dists = np.sqrt(d_sq)
        elevs = np.degrees(np.arctan2(flight_alt - r_alt, dists))
        in_el = np.abs(elevs) <= r["波束宽度_deg"]
        in_beam |= (in_det & in_az & in_el)

    # 3) 各类盲区
    blind_out_of_range = ~in_range                          # 探测范围外
    blind_beam = in_range & ~in_beam                        # 角度限制
    blind_terrain = in_range & in_beam & blind_all          # 地形遮挡
    blind_other = in_range & in_beam & ~blind_all           # 可见 (for stats)

    print(f"  探测范围外: {float(np.sum(blind_out_of_range)):.0f} 格点")
    print(f"  波束角度盲区: {float(np.sum(blind_beam)):.0f} 格点 "
          f"({float(np.sum(blind_beam))*res**2/1e6:.1f} km^2)")
    print(f"  地形遮挡盲区: {float(np.sum(blind_terrain)):.0f} 格点 "
          f"({float(np.sum(blind_terrain))*res**2/1e6:.1f} km^2)")
    print(f"  可见区域: {float(np.sum(blind_other)):.0f} 格点 "
          f"({float(np.sum(blind_other))*res**2/1e6:.1f} km^2)")

    return viewsheds, blind_all, blind_terrain, blind_beam, blind_out_of_range
