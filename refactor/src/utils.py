"""
常量与工具函数
"""
import numpy as np

# 地图尺寸 (m)
MAP_WIDTH = 60000
MAP_HEIGHT = 50000

# 分辨率 (m)
WORK_RES = 100   # 开发分辨率
FINAL_RES = 5    # 提交分辨率

# 无人机参数
UAV_SPEED = 200        # km/h
FLIGHT_ALT = 500       # m, 巡航高度
MIN_TURN_RADIUS = 200  # m

# 坦克参数
TANK_SPEED = 50  # km/h

# 雷达参数
RADAR_RANGE = 10000  # m, 探测距离
RADAR_FIRE = 8000    # m, 火力半径
RADAR_BEAM = 60      # deg, 波束宽度


def km(v):
    return v * 1000


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def bearing(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dx, dy)) % 360
