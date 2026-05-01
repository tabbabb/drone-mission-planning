"""
地形生成模块 — 概率分布驱动的自然地形

核心思路:
- 山峰高度用指数分布: 大量低丘(50-300m)，少量高山(>500m)，极少数高峰(>1000m)
- 山体尺寸用对数正态分布: 大多数中小型，少数大型山脉
- 位置用聚类+均匀混合: 山脉区聚类散布 + 全图均匀散落
- 坡度自然形成: power参数也随机，确保多样化山形
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from src.utils import MAP_WIDTH, MAP_HEIGHT, LOW_RES


def _ellipse(X, Y, cx, cy, a, b, theta_deg, peak, power=2):
    t = np.radians(theta_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)
    Xr = (X - cx) * cos_t + (Y - cy) * sin_t
    Yr = -(X - cx) * sin_t + (Y - cy) * cos_t
    d = np.sqrt((Xr / a)**2 + (Yr / b)**2)
    return peak * np.exp(-0.5 * d ** power)


def generate_dem(res=LOW_RES):
    cols = int(MAP_WIDTH // res)
    rows = int(MAP_HEIGHT // res)
    x = np.linspace(0, MAP_WIDTH, cols)
    y = np.linspace(0, MAP_HEIGHT, rows)
    X, Y = np.meshgrid(x, y)

    rng = np.random.RandomState(42)
    dem = np.zeros((rows, cols))

    # ---- 基底微起伏 ----
    for scale, amp in [(8000, 4), (4000, 2), (2000, 0.8)]:
        nc = max(int(MAP_WIDTH / scale), 3)
        nr = max(int(MAP_HEIGHT / scale), 3)
        from scipy.ndimage import zoom
        noise = zoom(rng.randn(nr, nc), (rows / nr, cols / nc), order=1)
        dem += noise[:rows, :cols] * amp

    # ================================================================
    # 概率分布定义
    # ================================================================

    # 1. 山脉集群中心 (4个主要山脉区)
    cluster_centers = [
        # (cx, cy, spread_radius, n_features)
        # 龙门 — 大型主山脉
        (28000, 12000, 8000, 35),
        (31000, 24000, 9000, 40),
        (34500, 40000, 7500, 30),
        # 龙门连接带
        (29000, 18000, 5000, 12),
        (32500, 32000, 5000, 12),
        # 西岭 — 中型
        (12000, 11000, 6000, 22),
        (13500, 27000, 6500, 24),
        (15500, 41500, 5500, 18),
        # 东屏 — 中小型
        (44500, 44000, 5500, 18),
        (49800, 29500, 6000, 20),
        (54800, 15500, 5500, 16),
        # 北岭 — 小型低缓
        (21000, 44200, 5000, 14),
        (32500, 45800, 5200, 15),
        (42000, 44500, 4500, 12),
    ]

    total = 0

    # ---- 生成集群山体 ----
    for cx, cy, spread, n_features in cluster_centers:
        for _ in range(n_features):
            # 位置: 以集群中心为均值的高斯散布
            px = rng.normal(cx, spread * 0.35)
            py = rng.normal(cy, spread * 0.35)
            px = np.clip(px, 0, MAP_WIDTH)
            py = np.clip(py, 0, MAP_HEIGHT)

            # 高度: 指数分布 — 大量低丘, 少数高山
            # scale=平均高度, 指数分布P(X>x)=exp(-x/scale)
            peak = rng.exponential(scale=180) + 20  # min~20m, mean~200m, tail to ~1000m+
            peak = min(peak, 1600)  # 截断极端值

            # 尺寸: 对数正态 — 多数中小, 少数大型
            # median ~ exp(7.2) ≈ 1340m, sigma控制散布
            size = rng.lognormal(mean=7.2, sigma=0.6)
            size = np.clip(size, 400, 8000)
            a = size
            b = size * rng.uniform(0.5, 0.95)  # 随机椭圆率

            # 旋转角
            theta = rng.uniform(-30, 30)

            # 形状参数: 多样化 — 低丘用低power(平缓), 高峰用高power(陡峭)
            if peak < 100:
                power = rng.uniform(1.3, 1.8)
            elif peak < 300:
                power = rng.uniform(1.5, 2.2)
            elif peak < 600:
                power = rng.uniform(1.8, 2.8)
            elif peak < 1000:
                power = rng.uniform(2.2, 3.5)
            else:
                power = rng.uniform(3.0, 4.5)  # 最高峰最陡 → 满足45°坡度

            dem += _ellipse(X, Y, px, py, a, b, theta, peak, power)
            total += 1

    # ---- 全图均匀散落小山丘 ----
    # 3000 km^2 地图, 每 100 km^2 约 2-3 个小丘
    n_scattered = 80
    for _ in range(n_scattered):
        px = rng.uniform(0, MAP_WIDTH)
        py = rng.uniform(0, MAP_HEIGHT)

        # 散落山丘更低更小
        peak = rng.exponential(scale=60) + 10
        peak = min(peak, 350)
        size = rng.lognormal(mean=6.8, sigma=0.5)
        size = np.clip(size, 250, 3500)
        a = size
        b = size * rng.uniform(0.5, 0.9)
        theta = rng.uniform(-45, 45)
        power = rng.uniform(1.2, 2.0)  # 低丘一律平缓

        dem += _ellipse(X, Y, px, py, a, b, theta, peak, power)
        total += 1

    print(f"  Total features: {total}")
    print(f"  Height distribution: exponential(scale=180) for clusters, exponential(scale=60) for scattered")
    print(f"  Size distribution: lognormal(mean=7.2, sigma=0.6)")

    dem = gaussian_filter(dem, sigma=0.6)

    # 统计
    above_0 = dem[dem > 5]
    if len(above_0) > 0:
        p10, p50, p90, p99 = np.percentile(above_0, [10, 50, 90, 99])
        print(f"  Elevation (terrain>5m): P10={p10:.0f}m P50={p50:.0f}m P90={p90:.0f}m P99={p99:.0f}m")
    print(f"  Max: {dem.max():.0f}m  Mean: {dem.mean():.0f}m")

    extent = (0, MAP_WIDTH, 0, MAP_HEIGHT)
    return dem, X, Y, extent


def get_mountain_labels():
    return [
        {"name": "龙门山脉",   "label_x": 30500, "label_y": 24000, "peak_elevation": 1600},
        {"name": "西岭山脉",   "label_x": 13500, "label_y": 27000, "peak_elevation": 1200},
        {"name": "东屏丘陵",   "label_x": 50000, "label_y": 30000, "peak_elevation": 900},
        {"name": "北岭",       "label_x": 31000, "label_y": 45000, "peak_elevation": 700},
    ]


def compute_slope(dem, res=LOW_RES):
    dy, dx = np.gradient(dem, res)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))


def validate_requirements(dem, res=LOW_RES):
    slope = compute_slope(dem, res)
    qualified = (slope >= 45) & (dem >= 500)
    return {
        "max_elevation": float(np.max(dem)),
        "max_slope": float(np.max(slope)),
        "mean_elevation": float(np.mean(dem)),
        "qualified_pixels": int(np.sum(qualified)),
        "qualified_area_km2": float(np.sum(qualified) * (res**2) / 1e6),
        "meets_requirements": np.max(dem) >= 500 and np.max(slope) >= 45,
    }
