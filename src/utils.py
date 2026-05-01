"""
通用工具函数 — 坐标转换、地理计算、常量定义
地图空间: 60km × 50km，坐标系原点在左下角 (0, 0)
"""
import numpy as np

# === 地图常量 ===
MAP_WIDTH = 60000   # m, X方向
MAP_HEIGHT = 50000  # m, Y方向
LOW_RES = 100       # m, 低分辨率开发网格
FINAL_RES = 5       # m, 最终提交分辨率
FLIGHT_ALTITUDE = 500  # m, 无人机巡航高度
MIN_TURN_RADIUS = 200  # m, 最小转弯半径

# === 无人机参数 ===
NUM_UAVS = 4
UAV_SPEED = 200  # km/h, 典型无人机巡航速度

# === 坦克参数 ===
TANK_SPEED = 50  # km/h

# === 地形参数 ===
NOISE_SCALE = 1500   # m, 噪声空间尺度
BASE_ROUGHNESS = 30  # m, 基底起伏幅度

# === 雷达参数 ===
RADAR_DETECTION_RANGE = 10000  # m, 10km
RADAR_FIRE_RANGE = 8000        # m, 8km
RADAR_BEAM_WIDTH = 60          # deg, 俯仰/航向均为±60°

# === 坐标转换 ===
def km_to_m(km):
    """公里转米"""
    return km * 1000

def m_to_km(m):
    """米转公里"""
    return m / 1000

def world_to_grid(x, y, res=LOW_RES):
    """世界坐标(m) → DEM网格索引"""
    return int(round(y / res)), int(round(x / res))

def grid_to_world(row, col, res=LOW_RES):
    """DEM网格索引 → 世界坐标(m) 返回中心点"""
    return col * res + res / 2, row * res + res / 2

def make_grid(res=LOW_RES):
    """生成世界坐标网格 (X, Y 二维数组)"""
    cols = int(MAP_WIDTH // res)
    rows = int(MAP_HEIGHT // res)
    x = np.linspace(0, MAP_WIDTH, cols)
    y = np.linspace(0, MAP_HEIGHT, rows)
    return np.meshgrid(x, y)

def distance(p1, p2):
    """两点欧氏距离 (m)"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def bearing(p1, p2):
    """p1 → p2 方位角 (deg, 0=北, 顺时针)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dx, dy)) % 360

def is_inside_polygon(point, polygon):
    """射线法判断点是否在多边形内"""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def slope_percent(elev_diff, dist):
    """计算坡度百分比"""
    if dist == 0:
        return 0
    return abs(elev_diff / dist)

def slope_deg(elev_diff, dist):
    """计算坡度角度"""
    return np.degrees(np.arctan(slope_percent(elev_diff, dist)))
