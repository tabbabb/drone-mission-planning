"""
无人机任务规划 — 主入口

步骤:
  step1  生成地形图
  step2  (待实现) 威胁建模
  step3  (待实现) 雷达盲区
  step4  (待实现) 综合地图
  step5  (待实现) 路径规划
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

from src.terrain import generate, compute_slope, stats
from src.utils import MAP_WIDTH, MAP_HEIGHT

OUTPUT = "output"
import os
os.makedirs(OUTPUT, exist_ok=True)


def _find_peaks(dem, min_height=400, min_distance_km=3):
    """找DEM局部极大值 → [(x_m, y_m, elevation), ...]"""
    res = 100  # m
    radius = int(min_distance_km * 1000 / res)
    size = radius * 2 + 1
    local_max = dem == maximum_filter(dem, size=size)
    peaks = local_max & (dem >= min_height)
    rows, cols = np.where(peaks)
    # 按高度降序，去重（相邻峰只保留最高）
    results = []
    for r, c in sorted(zip(rows, cols), key=lambda x: -dem[x[0], x[1]]):
        x = c * res + res / 2
        y = r * res + res / 2
        h = dem[r, c]
        # 检查是否与已保留的峰距离足够
        if all(np.sqrt((x - px)**2 + (y - py)**2) >= min_distance_km * 1000
               for px, py, _ in results):
            results.append((x, y, h))
        if len(results) >= 6:
            break
    return results


def step1_terrain():
    """步骤1: 生成地形图"""
    print("=" * 50)
    print("Step 1 — 地形生成")
    print("=" * 50)

    dem, X, Y = generate()
    stats(dem)
    slope = compute_slope(dem)
    peaks = _find_peaks(dem, min_height=400, min_distance_km=4)

    print("主要山峰坐标:")
    for x, y, h in peaks:
        print(f"  ({x/1000:.1f}, {y/1000:.1f}) km  {h:.0f}m")

    # --- 三合一图 ---
    fig = plt.figure(figsize=(20, 7))
    ext = (0, MAP_WIDTH / 1000, 0, MAP_HEIGHT / 1000)

    # —— 地形高程图 ——
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(dem, extent=ext, origin='lower', cmap='terrain', aspect='auto')
    ax1.set_title('数字高程模型 (DEM)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)')
    plt.colorbar(im1, ax=ax1, label='海拔 (m)', shrink=0.82)

    # 标注山峰坐标（偏移避免遮盖）
    offsets = [(-30, 30), (30, -30), (-30, -30), (30, 30), (-25, 25), (25, -25)]
    for (x, y, h), (dx, dy) in zip(peaks, offsets):
        ax1.plot(x / 1000, y / 1000, '^', color='white', markersize=8,
                 markeredgecolor='black', markeredgewidth=1)
        ax1.annotate(f'({x/1000:.1f}, {y/1000:.1f})  {h:.0f}m',
                     xy=(x / 1000, y / 1000), xytext=(dx, dy),
                     textcoords='offset points', fontsize=6.5, color='white',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#5C3317', alpha=0.8),
                     arrowprops=dict(arrowstyle='-', color='#5C3317', lw=0.8))

    # —— 坡度图 ——
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(slope, extent=ext, origin='lower', cmap='YlOrRd', aspect='auto', vmax=50)
    ax2.set_title('坡度图', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (km)'); ax2.set_ylabel('Y (km)')
    plt.colorbar(im2, ax=ax2, label='坡度 (°)', shrink=0.82)

    # —— 三维地形 ——
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    s = 8
    X3, Y3, D3 = X[::s, ::s] / 1000, Y[::s, ::s] / 1000, dem[::s, ::s]
    norm = plt.Normalize(D3.min(), D3.max())
    ax3.plot_surface(X3, Y3, D3, facecolors=plt.cm.terrain(norm(D3)),
                     linewidth=0, antialiased=True, alpha=0.9, shade=True)
    ax3.set_title('三维地形视图', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (km)'); ax3.set_ylabel('Y (km)'); ax3.set_zlabel('海拔 (m)')
    ax3.view_init(elev=40, azim=-55)

    plt.tight_layout()
    path = f'{OUTPUT}/step1_terrain.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {path}")
    return dem


if __name__ == '__main__':
    dem = step1_terrain()
