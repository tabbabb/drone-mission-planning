"""
威胁建模 — 防空系统、建筑群、禁飞区、动态威胁、环境因素
"""
import numpy as np
from .utils import RADAR_RANGE, RADAR_FIRE, RADAR_BEAM, FLIGHT_ALT, MAP_WIDTH, MAP_HEIGHT, WORK_RES


# ================================================================
# 1. 防空雷达系统 (>=2套)
# ================================================================
RADARS = [
    {
        "id": "R1",
        "型号": "AN/MPQ-65",
        "位置": (20000, 22000),       # (x, y) m
        "天线高度_m": 15,
        "探测距离_m": RADAR_RANGE,    # 10km
        "火力半径_m": RADAR_FIRE,     # 8km
        "波束宽度_deg": RADAR_BEAM,   # ±60°
        "主瓣朝向_deg": 90,           # 0=北, 90=东
    },
    {
        "id": "R2",
        "型号": "AN/TPS-75",
        "位置": (40000, 28000),
        "天线高度_m": 12,
        "探测距离_m": RADAR_RANGE,
        "火力半径_m": RADAR_FIRE,
        "波束宽度_deg": RADAR_BEAM,
        "主瓣朝向_deg": 270,          # 朝西
    },
]


def radar_mask(radar, dem, X, Y, flight_alt=FLIGHT_ALT, res=WORK_RES):
    """
    计算单部雷达可见区域 (简化圆 + 波束, 不考虑地形遮挡)
    返回: (detection_mask, fire_mask)
    """
    rx, ry = radar["位置"]
    cx = np.arange(dem.shape[1]) * res + res / 2
    cy = np.arange(dem.shape[0]) * res + res / 2
    Xg, Yg = np.meshgrid(cx, cy)

    dist = np.sqrt((Xg - rx)**2 + (Yg - ry)**2)

    # 方位角过滤
    bearing = np.degrees(np.arctan2(Xg - rx, Yg - ry)) % 360
    az_diff = (bearing - radar["主瓣朝向_deg"] + 180) % 360 - 180

    in_beam = np.abs(az_diff) <= radar["波束宽度_deg"]

    det = (dist <= radar["探测距离_m"]) & in_beam
    fire = (dist <= radar["火力半径_m"]) & in_beam

    return det, fire


# ================================================================
# 2. 高层建筑群 (>=1片, 相对高度>=100m)
# ================================================================
BUILDING_CLUSTER = {
    "名称": "临港商务区",
    "中心": (28000, 15000),
    "半径_X_m": 3000,
    "半径_Y_m": 2000,
    "相对高度_m": 120,
    "建筑列表": [
        (27500, 15000, 120),
        (28000, 15300, 130),
        (28500, 14800, 115),
        (27800, 14600, 125),
        (28200, 15400, 140),
        (27300, 15200, 110),
        (28300, 14500, 135),
        (27900, 15500, 120),
    ],
}


def is_in_building_cluster(x, y):
    bc = BUILDING_CLUSTER
    dx = (x - bc["中心"][0]) / bc["半径_X_m"]
    dy = (y - bc["中心"][1]) / bc["半径_Y_m"]
    return (dx**2 + dy**2) <= 1.0


# ================================================================
# 3. 禁飞区 (>=1个多边形)
# ================================================================
NO_FLY_ZONE = {
    "名称": "军事禁区",
    "顶点_m": [
        (5000, 35000),
        (15000, 38000),
        (14000, 45000),
        (6000, 43000),
        (3000, 39000),
    ],
}


def _ray_casting(x, y, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def is_in_no_fly_zone(x, y):
    return _ray_casting(x, y, NO_FLY_ZONE["顶点_m"])


# ================================================================
# 4. 动态威胁 — 移动坦克 (加分项)
# ================================================================
MOVING_TARGETS = [
    {
        "id": "T3",
        "名称": "坦克纵队A",
        "初始位置": (42000, 8000),
        "速度_kmh": 50,
        "航向_deg": 30,       # NNE
        "尺寸_m": (8, 3),
    },
    {
        "id": "T4",
        "名称": "坦克纵队B",
        "初始位置": (38000, 42000),
        "速度_kmh": 50,
        "航向_deg": 120,      # ESE
        "尺寸_m": (8, 3),
    },
]

FIXED_TARGETS = [
    {"id": "T1", "名称": "固定目标Alpha", "位置": (52000, 35000), "尺寸_m": (100, 20)},
    {"id": "T2", "名称": "固定目标Beta",  "位置": (48000, 12000), "尺寸_m": (100, 20)},
]

ALL_TARGETS = FIXED_TARGETS + MOVING_TARGETS


def predict_tank(tank, t_sec):
    """预测坦克 t_sec 秒后的位置"""
    v_ms = tank["速度_kmh"] * 1000 / 3600
    rad = np.radians(tank["航向_deg"])
    x = tank["初始位置"][0] + v_ms * t_sec * np.sin(rad)
    y = tank["初始位置"][1] + v_ms * t_sec * np.cos(rad)
    return (np.clip(x, 0, MAP_WIDTH), np.clip(y, 0, MAP_HEIGHT))


# ================================================================
# 5. 环境因素 — 风速风向云底高度 (加分项)
# ================================================================
WIND = {
    "风速_m_s": 8.5,       # m/s, 典型中空风速
    "风向_deg": 225,       # SW → NE
    "云底高度_m": 1200,    # 云底高, 影响飞行高度选择
    "能见度_km": 15,
}


def wind_effect(uav_speed_kmh):
    """简化风影响: 顺风/逆风修正地速"""
    uav_heading = 0  # 假设UAV朝东飞
    wind_dir = WIND["风向_deg"]
    wind_spd = WIND["风速_m_s"] * 3.6  # km/h
    # 顺风分量
    tailwind = wind_spd * np.cos(np.radians(wind_dir - uav_heading))
    return uav_speed_kmh + tailwind
