"""
地形生成 — 分布驱动的自然山脉
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from .utils import MAP_WIDTH, MAP_HEIGHT, WORK_RES


def _ellipse(X, Y, cx, cy, a, b, theta_deg, peak, power=2):
    t = np.radians(theta_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)
    Xr = (X - cx) * cos_t + (Y - cy) * sin_t
    Yr = -(X - cx) * sin_t + (Y - cy) * cos_t
    d = np.sqrt((Xr / a)**2 + (Yr / b)**2)
    return peak * np.exp(-0.5 * d ** power)


def generate(width=MAP_WIDTH, height=MAP_HEIGHT, res=WORK_RES, seed=42):
    cols = int(width // res)
    rows = int(height // res)
    x = np.linspace(0, width, cols)
    y = np.linspace(0, height, rows)
    X, Y = np.meshgrid(x, y)

    rng = np.random.RandomState(seed)
    dem = np.zeros((rows, cols))

    # 基底微起伏
    for scale, amp in [(8000, 4), (4000, 2), (2000, 0.8)]:
        nc = max(int(width / scale), 3)
        nr = max(int(height / scale), 3)
        noise = zoom(rng.randn(nr, nc), (rows / nr, cols / nc), order=1)
        dem += noise[:rows, :cols] * amp

    # 山脉集群
    w, h = width, height
    area = w * h / 1e6
    scale = np.sqrt(area / 300)

    clusters = [
        (w * 0.28, h * 0.24, 8000, int(35 * scale)),
        (w * 0.30, h * 0.10, 6000, int(15 * scale)),
        (w * 0.32, h * 0.46, 7000, int(22 * scale)),
        (w * 0.13, h * 0.22, 5500, int(18 * scale)),
        (w * 0.14, h * 0.43, 5500, int(16 * scale)),
        (w * 0.72, h * 0.44, 5000, int(16 * scale)),
        (w * 0.74, h * 0.24, 5000, int(14 * scale)),
        (w * 0.35, h * 0.44, 4500, int(12 * scale)),
        (w * 0.50, h * 0.43, 4000, int(10 * scale)),
    ]

    for cx, cy, spread, n in clusters:
        for _ in range(n):
            px = np.clip(rng.normal(cx, spread * 0.35), 0, width)
            py = np.clip(rng.normal(cy, spread * 0.35), 0, height)

            peak = rng.exponential(scale=55) + 8
            peak = min(peak, 420)

            size = rng.lognormal(mean=6.8, sigma=0.55)
            size = np.clip(size, 300, 5000)
            a, b = size, size * rng.uniform(0.5, 0.9)
            theta = rng.uniform(-30, 30)

            if peak < 40:
                power = rng.uniform(1.3, 1.8)
            elif peak < 100:
                power = rng.uniform(1.5, 2.2)
            elif peak < 200:
                power = rng.uniform(1.8, 2.8)
            elif peak < 340:
                power = rng.uniform(2.2, 3.5)
            else:
                power = rng.uniform(3.0, 4.5)

            dem += _ellipse(X, Y, px, py, a, b, theta, peak, power)

    # 散落低丘
    n_scattered = int(width * height / 2.5e7)
    for _ in range(n_scattered):
        px = rng.uniform(0, width)
        py = rng.uniform(0, height)
        peak = rng.exponential(scale=50) + 10
        peak = min(peak, 300)
        size = rng.lognormal(mean=6.5, sigma=0.45)
        size = np.clip(size, 200, 2500)
        a, b = size, size * rng.uniform(0.5, 0.9)
        theta = rng.uniform(-45, 45)
        power = rng.uniform(1.2, 2.0)
        dem += _ellipse(X, Y, px, py, a, b, theta, peak, power)

    dem = gaussian_filter(dem, sigma=0.6)
    return dem, X, Y


def compute_slope(dem, res=WORK_RES):
    dy, dx = np.gradient(dem, res)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))


def stats(dem):
    s = np.percentile(dem[dem > 5] if np.any(dem > 5) else dem, [10, 50, 90, 99])
    print(f"  Max: {dem.max():.0f}m  Mean: {dem.mean():.0f}m")
    print(f"  P10={s[0]:.0f}m  P50={s[1]:.0f}m  P90={s[2]:.0f}m  P99={s[3]:.0f}m")
