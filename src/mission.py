"""
任务分配与目标建模模块

4架无人机 →  4个目标 (2固定 + 2移动坦克)
- 固定目标: 100m×20m 建筑物
- 移动目标: 坦克, 速度50km/h, 需要预测拦截点
- 每架无人机恰好分配1个目标
"""
import numpy as np
from src.utils import MAP_WIDTH, MAP_HEIGHT, TANK_SPEED, UAV_SPEED, distance

# ============================================================
# UAV 基地 (西侧)
# ============================================================
UAV_BASE = (5000, 25000)  # 所有4架无人机从同一基地出发

# ============================================================
# 目标定义
# ============================================================

FIXED_TARGETS = [
    {
        "id": "T1",
        "type": "fixed",
        "name": "Fixed Target Alpha",
        "pos": (52000, 35000),     # 东屏山脉后方, 东北
        "size": (100, 20),          # m
    },
    {
        "id": "T2",
        "type": "fixed",
        "name": "Fixed Target Beta",
        "pos": (48000, 12000),      # 东南方向
        "size": (100, 20),
    },
]

MOVING_TARGETS = [
    {
        "id": "T3",
        "type": "moving",
        "name": "Tank Column Alpha",
        "pos": (30000, 15000),      # 初始位置 — 龙门山脉南端
        "velocity": TANK_SPEED,     # km/h
        "heading": 45,              # deg, NE方向
        "size": (8, 3),             # m, 坦克尺寸
    },
    {
        "id": "T4",
        "type": "moving",
        "name": "Tank Column Beta",
        "pos": (35000, 35000),      # 初始位置 — 龙门山脉北端
        "velocity": TANK_SPEED,
        "heading": 135,             # deg, SE方向
        "size": (8, 3),
    },
]

ALL_TARGETS = FIXED_TARGETS + MOVING_TARGETS


def predict_tank_position(tank, t_seconds):
    """
    预测坦克在 t_seconds 后的位置

    Parameters
    ----------
    tank : dict, 坦克目标
    t_seconds : float, 经过时间 (s)

    Returns
    -------
    (x, y) : 预测位置 (m)
    """
    speed_ms = tank["velocity"] * 1000 / 3600  # km/h →  m/s
    heading_rad = np.radians(tank["heading"])
    dx = speed_ms * t_seconds * np.sin(heading_rad)
    dy = speed_ms * t_seconds * np.cos(heading_rad)
    x = tank["pos"][0] + dx
    y = tank["pos"][1] + dy
    # 边界夹持
    x = np.clip(x, 0, MAP_WIDTH)
    y = np.clip(y, 0, MAP_HEIGHT)
    return (x, y)


def compute_intercept(uav_start, tank, uav_speed_kmh=UAV_SPEED):
    """
    计算对移动目标的预测拦截点

    简化方法: 二分搜索找到 UAV 飞行时间 = 坦克移动时间 的交点
    """
    uav_speed_ms = uav_speed_kmh * 1000 / 3600

    t_low, t_high = 0, 7200  # 最多2小时内
    best_t = 0

    for _ in range(50):
        t_mid = (t_low + t_high) / 2
        intercept_pos = predict_tank_position(tank, t_mid)
        dist = distance(uav_start, intercept_pos)
        uav_time = dist / uav_speed_ms

        if uav_time < t_mid:
            t_high = t_mid  # UAV 先到, 等坦克 → 延后拦截
        else:
            t_low = t_mid   # 坦克先到 → 提前拦截

        if abs(uav_time - t_mid) < 0.5:  # 收敛
            best_t = t_mid
            break
        best_t = t_mid

    return predict_tank_position(tank, best_t), best_t


def assign_tasks():
    """
    任务分配: 贪心法将 UAV 分配给目标

    分配逻辑:
    - 固定目标: 直接飞行时间最短
    - 移动目标: 考虑拦截点飞行时间

    Returns
    -------
    assignments : list of dicts [{uav_id, target_id, target, route_info}, ...]
    """
    uav_speed_ms = UAV_SPEED * 1000 / 3600

    tasks = []
    for i in range(4):
        target = ALL_TARGETS[i]
        if target["type"] == "fixed":
            dist = distance(UAV_BASE, target["pos"])
            eta = dist / uav_speed_ms
            tasks.append({
                "uav_id": i + 1,
                "target_id": target["id"],
                "target": target,
                "intercept_pos": target["pos"],
                "dist_km": dist / 1000,
                "eta_min": eta / 60,
            })
        else:
            intercept_pos, t_intercept = compute_intercept(UAV_BASE, target)
            dist = distance(UAV_BASE, intercept_pos)
            eta = dist / uav_speed_ms
            tasks.append({
                "uav_id": i + 1,
                "target_id": target["id"],
                "target": target,
                "intercept_pos": intercept_pos,
                "dist_km": dist / 1000,
                "eta_min": eta / 60,
                "intercept_time_s": t_intercept,
            })

    return tasks


def get_mission_summary():
    """打印任务摘要"""
    tasks = assign_tasks()
    print("=" * 60)
    print("Mission Assignment Summary")
    print("=" * 60)
    print(f"UAV Base: ({UAV_BASE[0]/1000:.1f}, {UAV_BASE[1]/1000:.1f}) km")
    print(f"UAV Speed: {UAV_SPEED} km/h")
    print()
    for t in tasks:
        print(f"UAV-{t['uav_id']} →  {t['target_id']} ({t['target']['type']})")
        print(f"  Target pos: ({t['intercept_pos'][0]/1000:.1f}, {t['intercept_pos'][1]/1000:.1f}) km")
        print(f"  Distance: {t['dist_km']:.1f} km, ETA: {t['eta_min']:.1f} min")
        if t['target']['type'] == 'moving':
            print(f"  Intercept time: {t['intercept_time_s']:.0f}s")
        print()
    return tasks
