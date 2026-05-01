"""
可视化模块 — 综合地图渲染、图例、比例尺、指北针

展示性为核心目标。配色、标注、叠加层均经过精心设计。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Polygon as MPolygon
import numpy as np
from src.utils import m_to_km, MAP_WIDTH, MAP_HEIGHT


# 配色方案 — 专业地图风格
COLORS = {
    "terrain_bg": "#F5F0E8",
    "contour": "#8B7355",
    "mountain_fill": "#8B4513",
    "radar_detection": "#FF6B6B",
    "radar_fire": "#FF4444",
    "radar_detection_alpha": 0.15,
    "radar_fire_alpha": 0.25,
    "building": "#708090",
    "no_fly": "#FFD700",
    "no_fly_alpha": 0.3,
    "route_primary": "#00CED1",
    "route_backup": "#FF8C00",
    "start_point": "#00FF00",
    "target_fixed": "#FF0000",
    "target_moving": "#FF69B4",
    "blind_zone": "#9370DB",
    "blind_zone_alpha": 0.25,
    "grid": "#CCCCCC",
    "water": "#4A90D9",
}


def setup_map_style():
    """配置全局地图样式"""
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = COLORS["terrain_bg"]


def add_north_arrow(ax, x=0.95, y=0.95, size=0.03):
    """添加指北针"""
    ax.annotate('N', xy=(x, y), xytext=(x, y - size * 1.5),
                fontsize=14, fontweight='bold', color='black',
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'),
                transform=ax.transAxes)


def add_scale_bar(ax, x=0.15, y=0.04, length_km=10):
    """添加比例尺 — 数据坐标假设为 km"""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    bar_x = xl + (xr - xl) * x
    bar_y = yb + (yt - yb) * y
    bar_width = length_km  # 数据坐标单位为 km

    ax.plot([bar_x, bar_x + bar_width], [bar_y, bar_y], 'k-', lw=4)
    ax.plot([bar_x, bar_x], [bar_y - 0.3, bar_y + 0.3], 'k-', lw=2)
    ax.plot([bar_x + bar_width, bar_x + bar_width], [bar_y - 0.3, bar_y + 0.3], 'k-', lw=2)
    ax.text(bar_x + bar_width / 2, bar_y + 0.8, f'{length_km} km',
            ha='center', va='bottom', fontsize=9, fontweight='bold')


def add_legend(ax, items, loc='lower right'):
    """添加图例"""
    handles = []
    for label, color, style in items:
        if style == 'patch':
            handles.append(mpatches.Patch(color=color, alpha=0.5, label=label))
        elif style == 'line':
            handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))
        elif style == 'marker':
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=color, markersize=8, label=label))
    ax.legend(handles=handles, loc=loc, fontsize=8, framealpha=0.9,
              edgecolor='gray', fancybox=True)


def plot_dem(ax, dem, extent_km, alpha=1.0, hillshade=True):
    """绘制DEM地形图（带山体阴影）extent_km = (left, right, bottom, top) in km"""
    pixel_res = MAP_WIDTH / dem.shape[1]  # actual pixel resolution in m

    if hillshade:
        dy, dx = np.gradient(dem, pixel_res)
        azimuth = 315
        altitude = 45
        az_rad = np.radians(azimuth)
        alt_rad = np.radians(altitude)

        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        shade = (np.cos(alt_rad) * np.cos(slope) +
                 np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
        shade = np.clip(shade, 0.3, 1.0)

        terrain_rgba = plt.cm.terrain(plt.Normalize()(dem))
        for i in range(3):
            terrain_rgba[:, :, i] *= shade
        ax.imshow(terrain_rgba, extent=extent_km, origin='lower', aspect='auto', alpha=alpha)
    else:
        ax.imshow(dem, extent=extent_km, origin='lower', cmap='terrain', aspect='auto', alpha=alpha)

    # 等高线 (extent_km 单位已是 km)
    levels = np.arange(100, 2000, 150)
    XX, YY = np.meshgrid(
        np.linspace(extent_km[0], extent_km[1], dem.shape[1]),
        np.linspace(extent_km[2], extent_km[3], dem.shape[0])
    )
    ax.contour(XX, YY, dem, levels=levels, colors=COLORS["contour"],
               linewidths=0.3, alpha=0.5)


def plot_radar_coverage(ax, radar_sites, detection_mask, fire_mask,
                        Xg, Yg, res=100):
    """绘制雷达覆盖范围"""
    from matplotlib.colors import ListedColormap

    extent = (0, MAP_WIDTH / 1000, 0, MAP_HEIGHT / 1000)

    # 探测范围
    det_cmap = ListedColormap([(1, 1, 1, 0), (*hex_to_rgb(COLORS["radar_detection"]), 0.15)])
    fire_cmap = ListedColormap([(1, 1, 1, 0), (*hex_to_rgb(COLORS["radar_fire"]), 0.25)])

    # 绘制雷达站点图标
    for radar in radar_sites:
        rx, ry = radar["position"]
        ax.plot(rx / 1000, ry / 1000, 's', color=COLORS["radar_fire"],
                markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                zorder=5)
        ax.annotate(radar["name"],
                    xy=(rx / 1000, ry / 1000),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                    zorder=6)

        # 探测距离圆
        det_circle = plt.Circle((rx / 1000, ry / 1000), radar["detection_range"] / 1000,
                                 fill=False, edgecolor=COLORS["radar_detection"],
                                 linestyle='--', linewidth=1, alpha=0.7)
        ax.add_patch(det_circle)

        # 火力距离圆
        fire_circle = plt.Circle((rx / 1000, ry / 1000), radar["fire_range"] / 1000,
                                  fill=False, edgecolor=COLORS["radar_fire"],
                                  linestyle='-', linewidth=1, alpha=0.5)
        ax.add_patch(fire_circle)


def plot_buildings(ax):
    """绘制建筑群"""
    from src.threats import BUILDING_CLUSTER
    bc = BUILDING_CLUSTER

    # 绘制建筑群范围椭圆
    ellipse = mpatches.Ellipse(
        (bc["center"][0] / 1000, bc["center"][1] / 1000),
        width=bc["radius_x"] * 2 / 1000,
        height=bc["radius_y"] * 2 / 1000,
        facecolor=COLORS["building"], alpha=0.3,
        edgecolor='black', linewidth=1, linestyle='-',
        zorder=3,
    )
    ax.add_patch(ellipse)

    # 绘制个别建筑
    bx = [b["x"] / 1000 for b in bc["buildings"]]
    by = [b["y"] / 1000 for b in bc["buildings"]]
    ax.scatter(bx, by, marker='^', color=COLORS["building"], s=20,
               edgecolors='black', linewidths=0.5, zorder=4)

    # 标注
    ax.annotate(bc["name"],
                xy=(bc["center"][0] / 1000, bc["center"][1] / 1000),
                fontsize=8, fontweight='bold', ha='center', va='center',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS["building"], alpha=0.8),
                zorder=5)


def plot_no_fly_zone(ax):
    """绘制禁飞区"""
    from src.threats import NO_FLY_ZONE
    nfz = NO_FLY_ZONE

    poly_km = [(x / 1000, y / 1000) for x, y in nfz["polygon"]]
    polygon = MPolygon(poly_km, closed=True,
                       facecolor=COLORS["no_fly"], alpha=COLORS["no_fly_alpha"],
                       edgecolor='#B8860B', linewidth=2, linestyle='--',
                       hatch='////', zorder=3)
    ax.add_patch(polygon)

    # 标注
    cx = np.mean([p[0] for p in poly_km])
    cy = np.mean([p[1] for p in poly_km])
    ax.annotate(nfz["name"],
                xy=(cx, cy), fontsize=10, fontweight='bold',
                ha='center', va='center', color='#8B0000',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                zorder=5)


def plot_routes(ax, routes, labels=None):
    """
    绘制航路

    Parameters
    ----------
    routes : list of arrays [(N,2), ...], 每条航路的 (x, y) 坐标
    labels : list of str, 标注
    """
    colors = [COLORS["route_primary"], COLORS["route_backup"],
              '#00FF7F', '#FF6347']
    for i, route in enumerate(routes):
        if route is None or len(route) == 0:
            continue
        rx = [p[0] / 1000 for p in route]
        ry = [p[1] / 1000 for p in route]
        color = colors[i % len(colors)]
        style = '-' if i % 2 == 0 else '--'
        lw = 2.5 if i % 2 == 0 else 2.0
        lab = labels[i] if labels and i < len(labels) else None
        ax.plot(rx, ry, style, color=color, lw=lw, label=lab, alpha=0.9, zorder=6)
        # 方向箭头
        if len(rx) > 2:
            mid_idx = len(rx) // 2
            ax.annotate('', xy=(rx[mid_idx + 1], ry[mid_idx + 1]),
                        xytext=(rx[mid_idx], ry[mid_idx]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2),
                        zorder=7)


def plot_mission_points(ax, uav_bases, targets):
    """
    绘制任务起止点

    Parameters
    ----------
    uav_bases : list of (x, y), 无人机出发点
    targets : list of dict, 目标 [{"pos": (x,y), "type": "fixed"/"moving", "id": str}, ...]
    """
    for i, base in enumerate(uav_bases):
        ax.plot(base[0] / 1000, base[1] / 1000, '^', color=COLORS["start_point"],
                markersize=12, markeredgecolor='black', markeredgewidth=1, zorder=7)
        ax.annotate(f'UAV{i+1}', xy=(base[0] / 1000, base[1] / 1000),
                    xytext=(5, -12), textcoords='offset points',
                    fontsize=7, fontweight='bold', color='darkgreen')

    for tgt in targets:
        pos = tgt["pos"]
        color = COLORS["target_fixed"] if tgt["type"] == "fixed" else COLORS["target_moving"]
        marker = 's' if tgt["type"] == "fixed" else 'D'
        ax.plot(pos[0] / 1000, pos[1] / 1000, marker=marker, color=color,
                markersize=12, markeredgecolor='black', markeredgewidth=1, zorder=7)
        ax.annotate(tgt["id"], xy=(pos[0] / 1000, pos[1] / 1000),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=7, fontweight='bold', color='darkred')


def compose_map(dem, extent, radar_sites, show_routes=None,
                show_points=None, title="Comprehensive Mission Map",
                output_path="comprehensive_map.png"):
    """
    综合地图主函数 — 整合所有图层

    Parameters
    ----------
    dem : np.ndarray, 地形数据
    extent : tuple, (left, right, bottom, top) in meters
    radar_sites : list of dict, 雷达站点
    show_routes : optional list of routes
    show_points : optional tuple of (uav_bases, targets)
    title : str
    output_path : str
    """
    setup_map_style()
    from src.threats import get_radar_zones

    det_mask, fire_mask, Xg, Yg = get_radar_zones()

    fig, ax = plt.subplots(figsize=(16, 12))

    # 背景
    ax.set_facecolor(COLORS["terrain_bg"])

    # 1. DEM地形
    plot_dem(ax, dem, (extent[0] / 1000, extent[1] / 1000,
                        extent[2] / 1000, extent[3] / 1000))

    # 2. 雷达覆盖
    plot_radar_coverage(ax, radar_sites, det_mask, fire_mask, Xg, Yg)

    # 3. 建筑群
    plot_buildings(ax)

    # 4. 禁飞区
    plot_no_fly_zone(ax)

    # 5. 航路
    if show_routes:
        plot_routes(ax, show_routes[0], show_routes[1] if len(show_routes) > 1 else None)

    # 6. 任务点
    if show_points:
        plot_mission_points(ax, show_points[0], show_points[1])

    # 7. 装饰元素
    add_north_arrow(ax)
    add_scale_bar(ax, length_km=10)

    # 图例
    legend_items = [
        ("Radar Detection", COLORS["radar_detection"], 'patch'),
        ("Radar Fire Zone", COLORS["radar_fire"], 'patch'),
        ("Building Cluster", COLORS["building"], 'patch'),
        ("No-Fly Zone", COLORS["no_fly"], 'patch'),
    ]
    if show_routes:
        legend_items.append(("Primary Route", COLORS["route_primary"], 'line'))
        legend_items.append(("Backup Route", COLORS["route_backup"], 'line'))
    if show_points:
        legend_items.append(("UAV Base", COLORS["start_point"], 'marker'))
        legend_items.append(("Fixed Target", COLORS["target_fixed"], 'marker'))
        legend_items.append(("Moving Target", COLORS["target_moving"], 'marker'))
    add_legend(ax, legend_items, loc='lower left')

    # 网格
    ax.grid(True, alpha=0.15, color=COLORS["grid"], linestyle='-', linewidth=0.5)

    ax.set_xlabel('X (km)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (km)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0, MAP_WIDTH / 1000)
    ax.set_ylim(0, MAP_HEIGHT / 1000)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Map saved: {output_path}")


def hex_to_rgb(hex_color):
    """Hex to (r, g, b) normalized"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
