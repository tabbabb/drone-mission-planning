"""
路径规划模块 — A* + 代价函数 + 航路平滑

代价组成:
- 地形: 超过安全高度的山体不可通行
- 雷达: 探测区内高代价, 火力区内极高代价
- 禁飞区: 不可通行
- 建筑群: 高代价
- 路径长度
"""
import numpy as np
import heapq
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from src.utils import (
    MAP_WIDTH, MAP_HEIGHT, LOW_RES, MIN_TURN_RADIUS, FLIGHT_ALTITUDE,
    distance,
)
from src.threats import (
    RADAR_SITES, BUILDING_CLUSTER, NO_FLY_ZONE,
    is_in_building_cluster, is_in_no_fly_zone, get_radar_zones,
)
from src.radar import compute_radar_viewshed


# 代价权重
COST_MOUNTAIN = 1e6      # 不可通行
COST_NO_FLY = 1e6        # 不可通行
COST_RADAR_FIRE = 5000   # 极高代价
COST_RADAR_DETECT = 800  # 高代价
COST_BUILDING = 3000     # 高代价
COST_BASE = 1.0          # 基础代价 (每米)

# 安全飞行高度以上多少米算"可飞越"
CLEARANCE = 100  # m, UAV飞在山上方至少100m


def build_cost_map(dem, viewshed=None, res=LOW_RES, flight_alt=FLIGHT_ALTITUDE):
    """
    构建综合代价地图

    Returns
    -------
    cost : np.ndarray (rows, cols), 每像素通行代价
    """
    rows, cols = dem.shape
    cost = np.full((rows, cols), COST_BASE, dtype=np.float64)

    # 1. 地形代价: UAV 飞行高度低于地形 + CLEARANCE 则不可通行
    mountain_mask = dem >= (flight_alt - CLEARANCE)
    cost[mountain_mask] = COST_MOUNTAIN

    # 2. 禁飞区
    from src.threats import get_no_fly_mask
    nfz_mask = get_no_fly_mask(res)
    cost[nfz_mask] = COST_NO_FLY

    # 3. 建筑群
    from src.threats import get_building_mask
    bld_mask = get_building_mask(res)
    cost[bld_mask] = np.maximum(cost[bld_mask], COST_BUILDING)

    # 4. 雷达覆盖
    det_mask, fire_mask, _, _ = get_radar_zones(res)
    cost[fire_mask] = np.maximum(cost[fire_mask], COST_RADAR_FIRE)
    cost[det_mask & ~fire_mask] = np.maximum(cost[det_mask & ~fire_mask], COST_RADAR_DETECT)

    # 如果传入viewshed, 盲区降低雷达代价
    if viewshed:
        for rid, vis in viewshed.items():
            # 雷达可见区保持高代价, 盲区降低
            pass  # cost已按简化圆设置, viewshed细节可在后续改进

    return cost


def astar_path(cost_map, start_world, goal_world, res=LOW_RES, allow_diag=True):
    """
    A* 网格路径搜索

    Parameters
    ----------
    cost_map : np.ndarray, 代价图
    start_world, goal_world : (x, y) 世界坐标 m
    res : 网格分辨率 m
    allow_diag : 允许8邻域

    Returns
    -------
    path : list of (x, y) or None
    """
    rows, cols = cost_map.shape

    start = (int(start_world[1] / res), int(start_world[0] / res))
    goal = (int(goal_world[1] / res), int(goal_world[0] / res))

    # 边界检查
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return None

    # 不可通行起点/终点
    if cost_map[start] >= COST_MOUNTAIN * 0.5:
        # 找最近可通行点
        start = _nearest_passable(cost_map, start)
        if start is None:
            return None
    if cost_map[goal] >= COST_MOUNTAIN * 0.5:
        goal = _nearest_passable(cost_map, goal)
        if goal is None:
            return None

    # A*
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diag:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    diag_factor = np.sqrt(2)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # 重建路径
            path = []
            while current in came_from:
                r, c = current
                path.append((c * res + res / 2, r * res + res / 2))
                current = came_from[current]
            r, c = start
            path.append((c * res + res / 2, r * res + res / 2))
            path.reverse()
            return path

        cr, cc = current
        for dr, dc in neighbors:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cell_cost = cost_map[nr, nc]
            if cell_cost >= COST_MOUNTAIN * 0.5:
                continue

            step_dist = res * (diag_factor if dr != 0 and dc != 0 else 1.0)
            move_cost = step_dist * cell_cost
            tentative_g = g_score[current] + move_cost

            neighbor = (nr, nc)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # heuristic: 欧氏距离 * COST_BASE
                h = distance((nc * res, nr * res),
                             (goal[1] * res, goal[0] * res)) * COST_BASE
                f = tentative_g + h
                heapq.heappush(open_set, (f, neighbor))

    return None


def _nearest_passable(cost_map, cell, max_radius=20):
    """在不可通行点附近搜索最近可通行点"""
    rows, cols = cost_map.shape
    for r in range(1, max_radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if cost_map[nr, nc] < COST_MOUNTAIN * 0.5:
                        return (nr, nc)
    return None


def generate_alternative_path(cost_map, start, goal, primary_path, res=LOW_RES):
    """
    生成备用路径: 在代价图中增加主路径附近的惩罚区域, 重新搜索
    使备用路径与主路径有显著差异
    """
    if primary_path is None:
        return astar_path(cost_map, start, goal, res)

    # 在primary_path周围增加代价
    alt_cost = cost_map.copy()
    penalty_radius_m = 3000  # 3km 排斥半径
    penalty_radius_cells = int(penalty_radius_m / res)

    for px, py in primary_path:
        pc = int(px / res)
        pr = int(py / res)
        r_min = max(0, pr - penalty_radius_cells)
        r_max = min(alt_cost.shape[0], pr + penalty_radius_cells + 1)
        c_min = max(0, pc - penalty_radius_cells)
        c_max = min(alt_cost.shape[1], pc + penalty_radius_cells + 1)
        for rr in range(r_min, r_max):
            for cc in range(c_min, c_max):
                d_cells = np.sqrt((rr - pr)**2 + (cc - pc)**2)
                if d_cells <= penalty_radius_cells:
                    penalty = COST_RADAR_DETECT * (1 - d_cells / penalty_radius_cells)
                    alt_cost[rr, cc] += penalty

    return astar_path(alt_cost, start, goal, res)


def smooth_path(path, min_turn_radius=MIN_TURN_RADIUS, num_points=200):
    """
    B-spline 平滑 + 曲率检查

    Returns
    -------
    smoothed : list of (x, y) 点数等于 num_points
    """
    if path is None or len(path) < 3:
        return path

    pts = np.array(path)
    n_orig = len(pts)

    # Cubic spline 参数化
    t = np.linspace(0, 1, n_orig)
    try:
        cs_x = CubicSpline(t, pts[:, 0])
        cs_y = CubicSpline(t, pts[:, 1])
    except Exception:
        return path

    t_smooth = np.linspace(0, 1, num_points)
    sx = cs_x(t_smooth)
    sy = cs_y(t_smooth)

    # 曲率检查与修正
    dx = np.gradient(sx)
    dy = np.gradient(sy)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
    turn_radius = 1.0 / (curvature + 1e-8)

    # 对曲率过大处做高斯平滑
    if np.any(turn_radius < min_turn_radius):
        sigma = 2.0
        for _ in range(3):
            sx = gaussian_filter1d(sx, sigma, mode='nearest')
            sy = gaussian_filter1d(sy, sigma, mode='nearest')
            dx = np.gradient(sx)
            dy = np.gradient(sy)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
            turn_radius = 1.0 / (curvature + 1e-8)
            if np.all(turn_radius >= min_turn_radius):
                break
            sigma *= 0.7

    return list(zip(sx, sy))


def plan_all_routes(dem, res=LOW_RES):
    """
    为所有4架UAV规划主用+备用航路

    Returns
    -------
    routes : dict, uav_id →  {primary: [(x,y),...], backup: [(x,y),...], target: dict}
    """
    from src.mission import assign_tasks, UAV_BASE

    tasks = assign_tasks()
    print("Building cost map...")
    cost_map = build_cost_map(dem, res=res)

    routes = {}
    for task in tasks:
        uid = task["uav_id"]
        goal = task["intercept_pos"]
        print(f"\nPlanning UAV-{uid} →  {task['target_id']}...")

        # 主用路径
        primary = astar_path(cost_map, UAV_BASE, goal, res=res)
        if primary:
            primary = smooth_path(primary)
            print(f"  Primary: {len(primary)} waypoints, "
                  f"length={_path_length(primary)/1000:.1f}km")
        else:
            print(f"  Primary: FAILED")

        # 备用路径
        backup = generate_alternative_path(cost_map, UAV_BASE, goal, primary, res=res)
        if backup:
            backup = smooth_path(backup)
            print(f"  Backup:  {len(backup)} waypoints, "
                  f"length={_path_length(backup)/1000:.1f}km")
        else:
            print(f"  Backup:  FAILED")

        routes[uid] = {
            "primary": primary,
            "backup": backup,
            "target": task["target"],
        }

    return routes


def _path_length(path):
    if path is None or len(path) < 2:
        return 0
    total = 0
    for i in range(1, len(path)):
        total += distance(path[i - 1], path[i])
    return total
