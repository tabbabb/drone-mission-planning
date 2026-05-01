"""
无人机任务规划 — 主入口

步骤:
  step1  生成地形图
  step2  威胁建模与可视化
  step3  雷达盲区分析
  step4  (待实现) 综合地图
  step5  (待实现) 路径规划
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import maximum_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

from src.terrain import generate, compute_slope, stats
from src.utils import MAP_WIDTH, MAP_HEIGHT
from src.threats import (RADARS, BUILDING_CLUSTER, NO_FLY_ZONE,
                          MOVING_TARGETS, FIXED_TARGETS, WIND, radar_mask)
from src.radar import compute_viewshed

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


def step2_threats(dem=None):
    """步骤2: 威胁建模与可视化"""
    print("=" * 50)
    print("Step 2 — 威胁建模")
    print("=" * 50)

    if dem is None:
        dem, X, Y = generate()
    else:
        _, X, Y = generate()  # regenerate for X,Y mesh

    ext = (0, MAP_WIDTH / 1000, 0, MAP_HEIGHT / 1000)

    # 打印威胁摘要（供报告引用）
    print("----- 威胁要素坐标 -----")
    for r in RADARS:
        rx, ry = r["位置"][0] / 1000, r["位置"][1] / 1000
        print(f"  {r['id']} {r['型号']}: ({rx:.1f}, {ry:.1f}) km  "
              f"天线{r['天线高度_m']}m  探测{r['探测距离_m']/1000:.0f}km  火力{r['火力半径_m']/1000:.0f}km  "
              f"波束±{r['波束宽度_deg']}°  朝向{r['主瓣朝向_deg']}°")
    bc = BUILDING_CLUSTER
    print(f"  建筑群 {bc['名称']}: 中心({bc['中心'][0]/1000:.1f}, {bc['中心'][1]/1000:.1f}) km  "
          f"范围({bc['半径_X_m']*2/1000:.1f}×{bc['半径_Y_m']*2/1000:.1f}) km  相对高度≥{bc['相对高度_m']}m  "
          f"共{len(bc['建筑列表'])}栋")
    nfz = NO_FLY_ZONE
    verts = ", ".join([f"({x/1000:.1f},{y/1000:.1f})" for x, y in nfz["顶点_m"]])
    print(f"  禁飞区 {nfz['名称']}: 顶点 [{verts}] km")
    for t in MOVING_TARGETS:
        print(f"  动态威胁 {t['名称']} ({t['id']}): 初始({t['初始位置'][0]/1000:.1f}, {t['初始位置'][1]/1000:.1f}) km  "
              f"速度{t['速度_kmh']}km/h  航向{t['航向_deg']}°")
    print(f"  环境: 风速{WIND['风速_m_s']}m/s  风向{WIND['风向_deg']}°  云底{WIND['云底高度_m']}m  能见度{WIND['能见度_km']}km")

    # --- 可视化 ---
    fig, ax = plt.subplots(figsize=(16, 12))

    # 地形底图
    ax.imshow(dem, extent=ext, origin='lower', cmap='terrain', aspect='auto', alpha=0.85, zorder=1)

    # ---- 雷达 ----
    # 生成雷达掩膜供后续路径规划使用
    radar_masks = {}
    for r in RADARS:
        det_mask, fire_mask = radar_mask(r, dem, X, Y)
        radar_masks[r["id"]] = {"det": det_mask, "fire": fire_mask}

        rx, ry = r["位置"][0] / 1000, r["位置"][1] / 1000
        det_r = r["探测距离_m"] / 1000
        fire_r = r["火力半径_m"] / 1000

        # 探测范围 (beam-limited polygon overlay)
        det_c = plt.Circle((rx, ry), det_r, fill=True,
                            facecolor='#FF6B6B', alpha=0.12,
                            edgecolor='#FF6B6B', linestyle='--', linewidth=1, zorder=2)
        ax.add_patch(det_c)
        # 火力范围
        fire_c = plt.Circle((rx, ry), fire_r, fill=True,
                             facecolor='#FF4444', alpha=0.20,
                             edgecolor='#FF4444', linestyle='-', linewidth=1.5, zorder=2)
        ax.add_patch(fire_c)
        # 雷达站点
        ax.plot(rx, ry, 's', color='#FF4444', markersize=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.annotate(f"{r['型号']} ({r['id']})\n探测{det_r:.0f}km 火力{fire_r:.0f}km",
                    xy=(rx, ry), xytext=(10, 10), textcoords='offset points',
                    fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    zorder=6)

    # ---- 建筑群 ----
    bc = BUILDING_CLUSTER
    bcx, bcy = bc["中心"][0] / 1000, bc["中心"][1] / 1000
    ax.add_patch(mpatches.Ellipse(
        (bcx, bcy), bc["半径_X_m"] * 2 / 1000, bc["半径_Y_m"] * 2 / 1000,
        facecolor='#708090', alpha=0.30, edgecolor='#333', linewidth=1.5, zorder=3))
    # 建筑图标
    bx = [b[0] / 1000 for b in bc["建筑列表"]]
    by = [b[1] / 1000 for b in bc["建筑列表"]]
    ax.scatter(bx, by, marker='^', color='#708090', s=20,
               edgecolors='black', linewidths=0.5, zorder=4)
    ax.annotate(f"高层建筑群\n相对高度≥{bc['相对高度_m']}m",
                xy=(bcx, bcy), fontsize=8, fontweight='bold', ha='center', va='center',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#708090', alpha=0.85),
                zorder=6)

    # ---- 禁飞区 ----
    nfz_km = [(x / 1000, y / 1000) for x, y in NO_FLY_ZONE["顶点_m"]]
    from matplotlib.patches import Polygon as MPolygon
    ax.add_patch(MPolygon(nfz_km, closed=True,
                           facecolor='#FFD700', alpha=0.25,
                           edgecolor='#8B0000', linewidth=2.5, linestyle='--',
                           hatch='////', zorder=3))
    cx_n = np.mean([p[0] for p in nfz_km])
    cy_n = np.mean([p[1] for p in nfz_km])
    ax.annotate(f"{NO_FLY_ZONE['名称']}",
                xy=(cx_n, cy_n), fontsize=10, fontweight='bold',
                ha='center', va='center', color='#8B0000',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFACD', alpha=0.85),
                zorder=6)

    # ---- 动态威胁: 坦克 ----
    tank_colors = ['#FF1493', '#FF4500']
    from src.threats import predict_tank
    for t, color in zip(MOVING_TARGETS, tank_colors):
        tx, ty = t["初始位置"][0] / 1000, t["初始位置"][1] / 1000

        # 1小时轨迹虚线
        end = predict_tank(t, 3600)
        ax.plot([tx, end[0] / 1000], [ty, end[1] / 1000],
                '--', color=color, alpha=0.4, linewidth=1.0, zorder=3)
        ax.plot(end[0] / 1000, end[1] / 1000, 'o', color=color, markersize=4,
                alpha=0.4, zorder=4)

        # 菱形标记
        ax.plot(tx, ty, 'D', color=color, markersize=13,
                markeredgecolor='black', markeredgewidth=1.5, zorder=6)
        ax.annotate(f"{t['名称']}  {t['速度_kmh']}km/h",
                    xy=(tx, ty), xytext=(12, -14), textcoords='offset points',
                    fontsize=7.5, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85),
                    zorder=7)

    # ---- 环境因素: 风 ----
    wx, wy = 52, 47  # km, 右上角
    ax.annotate(f"环境因素\n风速 {WIND['风速_m_s']}m/s  风向 {WIND['风向_deg']}°\n云底 {WIND['云底高度_m']}m  能见度 {WIND['能见度_km']}km",
                xy=(wx, wy), fontsize=7.5, color='#1E3A8A', fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                           edgecolor='#4169E1', linewidth=1),
                zorder=7)

    # ---- 图例 ----
    legend = [
        mpatches.Patch(color='#FF6B6B', alpha=0.3, label='雷达探测区 (10km)'),
        mpatches.Patch(color='#FF4444', alpha=0.4, label='雷达火力区 (8km)'),
        mpatches.Patch(color='#708090', alpha=0.4, label='高层建筑群 (≥100m)'),
        mpatches.Patch(color='#FFD700', alpha=0.3, label='禁飞区'),
        plt.Line2D([0], [0], marker='D', color='#FF1493', markerfacecolor='#FF1493',
                   markersize=9, label='动态威胁 — 坦克 (50km/h)'),
    ]
    ax.legend(handles=legend, loc='lower left', fontsize=8.5,
              framealpha=0.9, edgecolor='gray', fancybox=True, ncol=2)

    ax.set_xlabel('X (km)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (km)', fontsize=11, fontweight='bold')
    ax.set_title('威胁模型总览 — 防空系统 + 建筑群 + 禁飞区 + 动态威胁 + 环境因素',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0, MAP_WIDTH / 1000)
    ax.set_ylim(0, MAP_HEIGHT / 1000)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.10, color='#666', linewidth=0.5)

    plt.tight_layout()
    path = f'{OUTPUT}/step2_threats.png'
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {path}")
    return dem, radar_masks


def step3_blind_zones(dem=None):
    """步骤3: 雷达探测盲区分析"""
    print("=" * 50)
    print("Step 3 — 雷达盲区分析")
    print("=" * 50)

    if dem is None:
        dem, X, Y = generate()
    else:
        _, X, Y = generate()

    viewsheds, blind_all, blind_terrain, blind_beam, blind_out = compute_viewshed(dem)

    # --- 可视化: 三栏对比 ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ext = (0, MAP_WIDTH / 1000, 0, MAP_HEIGHT / 1000)

    titles = [
        f"R1 ({RADARS[0]['型号']}) 视域",
        f"R2 ({RADARS[1]['型号']}) 视域",
        "综合盲区分布"
    ]

    for ax, title, key in zip(axes, titles[:2], ["R1", "R2"]):
        ax.imshow(dem, extent=ext, origin='lower', cmap='terrain',
                  aspect='auto', alpha=0.6, zorder=1)

        vis = viewsheds[key]
        vis_rgba = np.zeros((*vis.shape, 4))
        vis_rgba[vis, :] = [0.0, 0.8, 0.2, 0.35]
        ax.imshow(vis_rgba, extent=ext, origin='lower', aspect='auto', zorder=2)

        r = RADARS[0] if key == "R1" else RADARS[1]
        rx, ry = r["位置"][0] / 1000, r["位置"][1] / 1000
        ax.plot(rx, ry, 's', color='red', markersize=8, markeredgecolor='black', zorder=4)
        ax.add_patch(plt.Circle((rx, ry), r["探测距离_m"] / 1000,
                                 fill=False, edgecolor='red', linestyle='--', linewidth=1))

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_xlim(0, MAP_WIDTH / 1000); ax.set_ylim(0, MAP_HEIGHT / 1000)
        ax.set_aspect('equal')

    # 综合盲区 (三类分明)
    ax3 = axes[2]
    ax3.imshow(dem, extent=ext, origin='lower', cmap='terrain',
               aspect='auto', alpha=0.6, zorder=1)
    # 1) 波束角度盲区 (探测圈内但角度不对 — 橙色)
    bb_rgba = np.zeros((*blind_beam.shape, 4))
    bb_rgba[blind_beam, :] = [1.0, 0.65, 0.0, 0.35]
    ax3.imshow(bb_rgba, extent=ext, origin='lower', aspect='auto', zorder=2)
    # 2) 地形遮挡盲区 (角度内但地形挡 — 深紫)
    bt_rgba = np.zeros((*blind_terrain.shape, 4))
    bt_rgba[blind_terrain, :] = [0.58, 0.0, 0.83, 0.40]
    ax3.imshow(bt_rgba, extent=ext, origin='lower', aspect='auto', zorder=3)

    for r in RADARS:
        rx, ry = r["位置"][0] / 1000, r["位置"][1] / 1000
        ax3.plot(rx, ry, 's', color='red', markersize=8, markeredgecolor='black', zorder=5)
        ax3.add_patch(plt.Circle((rx, ry), r["探测距离_m"] / 1000,
                                  fill=False, edgecolor='red', linestyle='--', linewidth=1))

    ax3.set_title('雷达探测盲区', fontsize=11, fontweight='bold')
    ax3.set_xlabel('X (km)'); ax3.set_ylabel('Y (km)')
    ax3.set_xlim(0, MAP_WIDTH / 1000); ax3.set_ylim(0, MAP_HEIGHT / 1000)
    ax3.set_aspect('equal')

    leg = [
        mpatches.Patch(color='#FFA500', alpha=0.35, label='波束角度盲区'),
        mpatches.Patch(color='purple', alpha=0.40, label='地形遮挡盲区'),
        mpatches.Patch(color='#00CC33', alpha=0.35, label='雷达可视区'),
    ]
    ax3.legend(handles=leg, loc='lower left', fontsize=8)

    plt.tight_layout()
    path = f'{OUTPUT}/step3_blind_zones.png'
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {path}")
    return viewsheds, blind_terrain, blind_beam


if __name__ == '__main__':
    dem = step1_terrain()
    _, radar_masks = step2_threats(dem)
    step3_blind_zones(dem)
