# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

无人机任务规划课程作业（Drone Mission Planning Coursework），分为两个阶段：

- **作业1**：数字地图生成 — 构建包含地形、威胁、禁飞区的60km×50km综合数字地图
- **作业2**：多无人机路径规划 — 4架无人机对4个目标（2固定+2移动坦克50km/h）进行任务分配与航迹规划

作业1的输出是作业2的输入基础，所有数据结构需兼容两阶段复用。

## 技术选型

- **语言**：Python 3
- **核心依赖**：numpy, scipy, matplotlib, shapely, rasterio（DEM读写）
- **规划算法**（作业2）在遗传算法、蚁群算法、粒子群算法中选取
- **展示**：matplotlib 静态图 + 可选的交互式HTML（folium/plotly）

## 关键参数

| 参数 | 值 |
|------|-----|
| 地图尺寸 | 60km × 50km |
| DEM分辨率 | 最终5m（开发阶段用低分辨率加速迭代） |
| 山峰 | ≥2座，改为山脉形态，海拔≥500m，坡度≥45°，可适当放大尺寸以增强展示效果 |
| 防空系统 | ≥2套，探测距离10km，火力半径8km，雷达波束±60°（俯仰和航向） |
| 高层建筑群 | ≥1片，相对高度≥100m |
| 禁飞区 | ≥1个多边形 |
| 无人机数量 | 4架，每架分配1个目标 |
| 航路要求 | 每架≥2条（1主+1备），曲率连续，转弯半径≥200m |
| 探测盲区 | 基于500m飞行高度计算三维视域，标注地形遮挡盲区 |

## 开发策略

1. **先低分辨率后高分辨率** — 开发和调试阶段用低分辨率DEM（如100m），所有算法验证通过后再生成5m分辨率最终产物
2. **山脉代替孤立山峰** — 将山峰合并为山脉地形，增大视觉冲击力，同时保证技术参数符合要求
3. **展示优先** — 可视化代码需包含图例、比例尺、指北针、威胁标注、航路叠加，配色方案注重区分度和美观
4. **代码分层** — 地形生成、威胁建模、雷达分析、路径规划各自独立模块，通过统一数据接口衔接

## 输出物清单

- [ ] `dem_lowres.tif` — 低分辨率DEM（开发用）
- [ ] `dem_5m.tif` — 5m分辨率DEM（最终提交）
- [ ] `threats.geojson` — 威胁要素矢量图层
- [ ] `blind_zones.geojson` — 雷达盲区矢量图层
- [ ] `comprehensive_map.png` — 综合地图（作业1最终产出）
- [ ] `route_plan.png` — 带航路叠加的综合展示图（作业2最终产出）
- [ ] `technical_report.md` — 技术说明文档
- [ ] 各模块的演示脚本（`demo_*.py`）

## 进度通知

使用 Bark 推送进行关键节点通知（通过 `mcp__bark__notify` 工具）。触发时机：
- 每个子任务完成时
- 遇到重大阻塞或需要决策时
- 最终产物生成时

## 目录结构

```
homework/
├── CLAUDE.md
├── dem_lowres.tif          # 低分辨率DEM（开发迭代用）
├── dem_5m.tif              # 5m分辨率DEM（最终提交）
├── threats.geojson         # 威胁要素矢量图层
├── blind_zones.geojson     # 雷达盲区矢量图层
├── comprehensive_map.png   # 作业1综合地图
├── route_plan.png          # 作业2航路规划图
├── technical_report.md     # 技术说明文档
├── src/
│   ├── terrain.py          # 地形生成（山脉、DEM）
│   ├── threats.py          # 威胁建模（防空系统、建筑群、禁飞区）
│   ├── radar.py            # 雷达探测与盲区分析
│   ├── visualization.py    # 可视化工具（地图渲染、图例、比例尺等）
│   ├── mission.py          # 任务分配与目标建模
│   ├── planner.py          # 路径规划算法
│   ├── route_smoothing.py  # 航路平滑与曲率处理
│   └── utils.py            # 坐标转换、通用工具
├── demo_terrain.py         # 地形生成演示
├── demo_threats.py         # 威胁要素演示
├── demo_radar.py           # 雷达分析演示
├── demo_comprehensive.py   # 综合地图演示（作业1最终）
└── demo_mission.py         # 路径规划演示（作业2最终）
```
