"""
威胁建模模块 — 防空系统、高层建筑群、禁飞区

威胁布局策略：
- 雷达部署在山间通道关键位置，封锁天然缺口
- 建筑群位于通道南端，进一步压缩可选路径
- 禁飞区封锁北部通道，迫使无人机必须在复杂约束下规划航路
"""
import numpy as np
from src.utils import (
    MAP_WIDTH, MAP_HEIGHT,
    RADAR_DETECTION_RANGE, RADAR_FIRE_RANGE, RADAR_BEAM_WIDTH,
    is_inside_polygon, distance,
)

# ============================================================
# 1. 防空雷达系统 (2套)
# ============================================================

RADAR_SITES = [
    {
        "id": "R1",
        "name": "AN/MPQ-65",
        "position": (20000, 22000),  # (x, y) m — 西岭与龙门之间的通道中部
        "antenna_height": 15,          # m, 天线架设高度
        "detection_range": RADAR_DETECTION_RANGE,  # 10km
        "fire_range": RADAR_FIRE_RANGE,            # 8km
        "beam_azimuth": 60,            # deg, ±60° azimuth
        "beam_elevation": 60,          # deg, ±60° elevation
        "facing": 90,                  # deg, 主瓣朝向 (0=N, 90=E)
    },
    {
        "id": "R2",
        "name": "AN/TPS-75",
        "position": (40000, 28000),    # 龙门与东屏之间的通道
        "antenna_height": 12,
        "detection_range": RADAR_DETECTION_RANGE,
        "fire_range": RADAR_FIRE_RANGE,
        "beam_azimuth": 60,
        "beam_elevation": 60,
        "facing": 270,                 # deg, 朝西
    },
]


def get_radar_coverage(radar, grid_points, dem, X, Y, flight_alt=500):
    """
    计算给定雷达对指定网格点的探测盲区

    Parameters
    ----------
    radar : dict, 雷达参数
    grid_points : list of (x, y), 待检查的点
    dem : np.ndarray, 地形高程
    X, Y : np.ndarray, 世界坐标网格
    flight_alt : float, 目标飞行高度 (m)

    Returns
    -------
    visible_mask : np.ndarray, bool — True=可探测
    """
    rx, ry = radar["position"]
    r_alt = radar["antenna_height"]

    visible = np.zeros(len(grid_points), dtype=bool)

    for i, (gx, gy) in enumerate(grid_points):
        dist = distance((rx, ry), (gx, gy))
        if dist > radar["detection_range"]:
            visible[i] = False
            continue

        # 检查方位角是否在波束范围内
        b = np.degrees(np.arctan2(gx - rx, gy - ry)) % 360  # 0=N, CW
        az_diff = (b - radar["facing"] + 180) % 360 - 180
        if abs(az_diff) > radar["beam_azimuth"]:
            visible[i] = False
            continue

        # 检查仰角是否在波束范围内
        elev_angle = np.degrees(np.arctan2(flight_alt - r_alt, dist))
        if abs(elev_angle) > radar["beam_elevation"]:
            visible[i] = False
            continue

        # Line-of-sight: 检查地形遮挡
        # 在雷达和目标之间采样若干个点
        n_samples = max(2, int(dist / 200))  # 每200m采样一次
        sample_x = np.linspace(rx, gx, n_samples)[1:-1]  # 去掉起点和终点
        sample_y = np.linspace(ry, gy, n_samples)[1:-1]

        # 插值获取采样点地形高度
        from scipy.ndimage import map_coordinates
        xs_norm = (sample_x - X[0, 0]) / (X[0, 1] - X[0, 0]) if X.shape[1] > 0 else sample_x
        ys_norm = (sample_y - Y[0, 0]) / (Y[1, 0] - Y[0, 0]) if Y.shape[0] > 0 else sample_y

        # 用最近邻法获取地形高度（简化但有效）
        res = X[0, 1] - X[0, 0]
        col_idx = np.clip((sample_x / res).astype(int), 0, dem.shape[1] - 1)
        row_idx = np.clip((sample_y / res).astype(int), 0, dem.shape[0] - 1)
        terrain_heights = dem[row_idx, col_idx]

        # LOS: 雷达与目标之间的视线方程
        los_heights = r_alt + (flight_alt - r_alt) * np.linspace(0, 1, n_samples)[1:-1]

        # 如果任何采样点地形 >= 视线高度，则被遮挡
        if np.any(terrain_heights >= los_heights):
            visible[i] = False
        else:
            visible[i] = True

    return visible


# ============================================================
# 2. 高层建筑群 (1片)
# ============================================================

BUILDING_CLUSTER = {
    "name": "临港商务区",
    "center": (28000, 15000),       # m, 建筑群中心 — 西岭-龙门通道南端
    "radius_x": 3000,                # m, 椭圆形区域半长轴(X)
    "radius_y": 2000,                # m, 椭圆形区域半短轴(Y)
    "relative_height": 120,          # m, 建筑相对高度≥100m
    "buildings": [
        {"x": 27500, "y": 15000, "h": 120},
        {"x": 28000, "y": 15300, "h": 130},
        {"x": 28500, "y": 14800, "h": 115},
        {"x": 27800, "y": 14600, "h": 125},
        {"x": 28200, "y": 15400, "h": 140},
        {"x": 27300, "y": 15200, "h": 110},
        {"x": 28300, "y": 14500, "h": 135},
        {"x": 27900, "y": 15500, "h": 120},
    ],
}


def is_in_building_cluster(x, y):
    """检查点是否在建筑群范围内"""
    bc = BUILDING_CLUSTER
    dx = (x - bc["center"][0]) / bc["radius_x"]
    dy = (y - bc["center"][1]) / bc["radius_y"]
    return (dx**2 + dy**2) <= 1.0


# ============================================================
# 3. 禁飞区 (1个多边形)
# ============================================================

NO_FLY_ZONE = {
    "name": "军事禁区",
    "polygon": [
        (5000, 35000),
        (15000, 38000),
        (14000, 45000),
        (6000, 43000),
        (3000, 39000),
    ],
}


def is_in_no_fly_zone(x, y):
    """检查点是否在禁飞区内"""
    return is_inside_polygon((x, y), NO_FLY_ZONE["polygon"])


# ============================================================
# 综合威胁评估
# ============================================================

def classify_zone(x, y):
    """
    综合威胁分类，返回区域类型字符串
    用于地图渲染和路径代价计算
    """
    if is_in_no_fly_zone(x, y):
        return "no_fly"
    if is_in_building_cluster(x, y):
        return "building"
    # 雷达和火力范围需在路径规划中动态计算
    return "free"


def get_radar_zones(res=100):
    """
    生成雷达覆盖区的栅格化掩膜 (简化版，不考虑地形遮挡)
    用于快速可视化预览
    """
    cols = int(MAP_WIDTH // res)
    rows = int(MAP_HEIGHT // res)
    detection_mask = np.zeros((rows, cols), dtype=bool)
    fire_mask = np.zeros((rows, cols), dtype=bool)

    x_coords = np.arange(cols) * res + res / 2
    y_coords = np.arange(rows) * res + res / 2
    Xg, Yg = np.meshgrid(x_coords, y_coords)

    for radar in RADAR_SITES:
        rx, ry = radar["position"]
        d = np.sqrt((Xg - rx)**2 + (Yg - ry)**2)
        detection_mask |= (d <= radar["detection_range"])
        fire_mask |= (d <= radar["fire_range"])

    return detection_mask, fire_mask, Xg, Yg


def get_building_mask(res=100):
    """生成建筑群栅格掩膜"""
    cols = int(MAP_WIDTH // res)
    rows = int(MAP_HEIGHT // res)
    mask = np.zeros((rows, cols), dtype=bool)
    x_coords = np.arange(cols) * res + res / 2
    y_coords = np.arange(rows) * res + res / 2

    for r in range(rows):
        for c in range(cols):
            mask[r, c] = is_in_building_cluster(x_coords[c], y_coords[r])
    return mask


def get_no_fly_mask(res=100):
    """生成禁飞区栅格掩膜"""
    cols = int(MAP_WIDTH // res)
    rows = int(MAP_HEIGHT // res)
    mask = np.zeros((rows, cols), dtype=bool)
    x_coords = np.arange(cols) * res + res / 2
    y_coords = np.arange(rows) * res + res / 2

    for r in range(rows):
        for c in range(cols):
            mask[r, c] = is_in_no_fly_zone(x_coords[c], y_coords[r])
    return mask
