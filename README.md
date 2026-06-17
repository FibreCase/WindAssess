# WindAccess - 风资源评估工具

基于 Python 的风资源测量数据处理与可视化分析工具，支持**雷达测风**（SODAR/LIDAR）和**测风塔**两种数据源，用于风电场选址评估。

## 功能概览

- **数据导入** — 自动识别并解析雷达与测风塔 CSV 数据文件
- **质量控制 (QC)** — 多轮次自动化质控流程，标记并剔除异常数据：
  - **范围检验**：剔除物理上不可能的值（如风速 >75 m/s、风向超出 0-360°）
  - **相关性检验**：湍流强度 >1、相邻高度间垂直风速差 >15 m/s、风向差 >120°
  - **趋势检验**：连续 6 个读数无变化的平值检测、突变 >20 m/s 的尖峰检测
  - **缺失检验**：风速或风向为 NaN 的记录
- **可用率统计** — 计算并输出原始数据与质控后各高度的数据可用率
- **可视化图表** — 生成 5 类专业分析图表（PNG，300 DPI）：
  - **Weibull 分布拟合**：各高度风速概率分布
  - **日变化曲线**：逐时风速均值、标准差、极值包络
  - **密度日变化**：基于理想气体定律由气温气压推算空气密度
  - **风切变日变化**：幂律模型计算相邻高度间风切变指数
  - **风玫瑰图**：16 方位风向频率玫瑰与风能密度玫瑰
- **数据导出** — 输出质控后的 CSV 文件供后续分析

## 项目结构

```
WindAccess/
├── main.py                     # 程序入口，串联完整流水线
├── pyproject.toml              # 项目配置与依赖声明
├── data/
│   ├── radar.csv               # 雷达测风数据
│   └── tower.csv               # 测风塔数据
├── src/
│   ├── data_file.py            # CSV 导入/导出工具
│   ├── chart/
│   │   ├── weibull_plot.py     # Weibull 分布拟合与绘图
│   │   ├── daily_variation.py  # 逐时风速统计与日变化曲线
│   │   ├── density_variation.py# 空气密度日变化（理想气体定律）
│   │   ├── shear_variation.py  # 风切变指数日变化（幂律模型）
│   │   └── wind_rose.py        # 风向玫瑰与风能玫瑰
│   ├── radar/
│   │   ├── qc_filter.py        # 雷达 QC 流水线（40-200m，步长 5m）
│   │   └── qc_cat.py           # 雷达数据可用率分类
│   └── tower/
│       ├── qc_filter.py        # 测风塔 QC 流水线（2/5/10/20/50/80m）
│       └── qc_cat.py           # 测风塔数据可用率分类
├── test/
│   ├── test_data_file.py       # 导入/导出单元测试
│   ├── test_radar_qc.py        # 雷达 QC 单元测试
│   └── test_tower_qc.py        # 测风塔 QC 单元测试
└── result/                     # 输出目录
    ├── qc_radar.csv            # 质控后雷达数据
    ├── qc_tower.csv            # 质控后测风塔数据
    └── chart/
        ├── radar/              # 雷达图表（按类型分子目录）
        └── tower/              # 测风塔图表（按类型分子目录）
```

## 环境要求

- **Python** >= 3.13
- **包管理器**：[uv](https://docs.astral.sh/uv/)

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd WindAccess

# 安装依赖
uv sync
```

## 使用方法

将数据文件放入 `data/` 目录后直接运行：

```bash
uv run python main.py
```

程序将依次执行：

1. 导入雷达与测风塔 CSV 数据
2. 对两组数据分别执行质量控制
3. 输出各高度数据可用率统计
4. 导出质控后 CSV 至 `result/`
5. 生成全部 5 类图表至 `result/chart/`

> 数据路径和输出路径均在 `main.py` 中硬编码，如需修改请直接编辑源码。

## 运行测试

```bash
uv run pytest
```

## 依赖

| 库 | 版本 | 用途 |
|---|---|---|
| pandas | >= 3.0.3 | 数据处理与 CSV 读写 |
| matplotlib | >= 3.11.0 | 图表绑定（子图、极坐标、直方图） |
| scipy | >= 1.17.1 | Weibull 分布拟合 |
| pytest | >= 9.1.0 | 单元测试（开发依赖） |

## 输出示例

程序运行后在 `result/chart/` 下生成约 44 张 PNG 图表：

| 图表类型 | 目录 | 说明 |
|---|---|---|
| Weibull 分布 | `weibull/` | 各高度风速概率密度拟合（分幅子图 + 叠加对比） |
| 日变化曲线 | `daily_variation/` | 逐时平均风速及标准差/极值包络带 |
| 密度日变化 | `density_variation/` | 逐时空气密度变化（由气温气压推算） |
| 风切变日变化 | `shear_variation/` | 相邻高度间逐时风切变指数（参考线 α=0.143） |
| 风玫瑰图 | `wind_rose/` | 16 方位风向频率玫瑰 + 风能密度玫瑰 |

每种图表均提供**单高度分幅图**和**多高度叠加图**两种形式。

## 数据格式约定

### 雷达数据

- 测量高度：40m ~ 200m（每 5m 一个层级，共 33 个高度）
- 列名格式：`Wind Speed{h}m`、`Wind Direction{h}m`、`Vertical Wind Speed{h}m`、`Wind Speed Std{h}m`

### 测风塔数据

- 测量高度：2m、5m、10m、20m、50m、80m
- 列名格式：`Avg Wind Speed @ {h}m [m/s]`、`Avg Wind Direction @ {h}m [deg]`、`Avg Wind Speed (std dev) @ {h}m [m/s]`
- 附加气象参数：温度（2m/50m/80m）、气压（`Station Pressure [mBar]`）

## 许可证

本项目未声明开源许可证，如需使用请联系作者。
