# WindAssess — 风资源评估工具

[English](README.md) | 中文

> 阅读 [最终报告](REPORT.md)

基于 Python 的风资源测量数据处理与评估工具，支持**声雷达（SODAR）**与
**常规测风塔**两类数据源，按 **NB/T 31147—2018**《风电场工程风能资源测量与
评估技术规范》的检验框架实现完整的质量控制与风资源参数分析，用于风电场
选址评估。

```
data/{radar,tower}.csv ──▶ QC 流水线（四道检验）──▶ result/qc_*.csv
                                            └─────▶ result/chart/{radar,tower}/（5 类图表）
```

## 数据源

| | 声雷达（radar） | 测风塔（tower） |
|---|---|---|
| 地区 | 西藏高原（复杂地形） | 常规地形（平原/丘陵） |
| 高度 | 40–200 m，5 m 一层，**33 层** | 2/5/10/20/50/80 m，**6 层** |
| 记录数 | 约 10,718 条 × 10 min | 约 52,950 条 × 10 min |
| 附加参数 | 垂直风速、单站温/压/湿 | 3 高度气温、站压、相对湿度 |

两类数据在观测手段、地形条件与垂直覆盖上差异显著，工具对两者执行
**结构对称的流水线**，并支持横向对比——`REPORT.md` 给出了一次真实运行的
完整对比分析（理论 + 结果）。

## 功能概览

- **多阶段质量控制（QC）**，逐高度分配 QC 编码（首个命中的检验生效）：
  - **范围检验**（编码 1）— 物理不可能值：风速超出 `[0, 75] m/s`、风向超出
    `[0°, 360°)`、垂直风速超出 `[-10, 10] m/s`、风速标准差超出 `[0, 20] m/s`
  - **缺测检验**（编码 5）— 风速/风向 NaN；在层间比较**之前**隔离，
    避免 NaN 传播
  - **相关性检验**（编码 2）— 湍流强度 `TI = σ/V` 超出 `[0, 1]`；相邻高度
    风速差 > **15 m/s**；相邻高度风向差 > **120°**（角度环绕感知：
    350° 与 10° 差 20° 而非 340°）
  - **趋势检验**（编码 3）— *平线*：连续 6 条读数完全相同（60 min 窗口，
    3 位小数舍入）；*突变*：相邻变化 > **20 m/s**
  - **廓线一致性检验**（编码 4）— 已实现（统计层间风速廓线一阶差分符号
    变化次数，标记不合理的频繁振荡），当前版本在主流程中**禁用**
    （33 层 × 万级行的计算开销过大）
- **不插补** — 被标记样本仅置 NaN，按实测数据报告，符合评估期统计规范
- **有效数据完整率**，按 NB/T 31147—2018 §5.2.10
  （`γ = (Rₑ − Rᵢ − R_w)/Rₑ`）；另提供**分步可用率追踪**，量化每道检验
  在各高度分别剔除了多少数据
- **风资源参数**（每类均输出逐高度分幅图 + 多高度叠加图，PNG 300 DPI）：
  - **Weibull 分布** — 两参数 MLE 拟合（`scipy.stats.weibull_min`，
    `floc=0`），给出形状参数 `k` / 尺度参数 `c`
  - **日变化** — 逐时风速均值 ± 标准差及极值包络
  - **空气密度** — 理想气体状态方程 `ρ = P/(R·T)`，由实测气温气压推算
  - **风切变** — 相邻高度幂律指数 `α = ln(V₂/V₁)/ln(h₂/h₁)`，带 1/7
    参考线（α = 0.143）
  - **风玫瑰图** — 16 方位（22.5°）风向频率玫瑰 + 风能密度玫瑰
    （`½ρV³`，标准密度 1.225 kg/m³）
- **单元测试**覆盖两类数据源的 QC 核心逻辑（范围/相关性/趋势）

## 快速开始

环境要求：**Python ≥ 3.13**，[uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/FibreCase/WindAssess.git
cd WindAssess
uv sync

# 将数据文件放入 data/
#   data/radar.csv   （声雷达）
#   data/tower.csv   （测风塔）

uv run python main.py
```

`main.py` 对两类数据依次执行完整流水线：

1. 导入 `data/radar.csv`、`data/tower.csv`（时间列自动解析）
2. 分别执行 QC + 分步可用率追踪
3. 导出质控后数据 → `result/qc_radar.csv`、`result/qc_tower.csv`
   （另输出 `result/qc_*_rate.csv` 分步可用率表）
4. 生成全部 5 类图表 → `result/chart/{radar,tower}/`

> 输入/输出路径硬编码在 `main.py` 中，如数据位置不同请直接修改源码。

### 运行测试

```bash
uv run pytest
```

## 项目结构

```
WindAssess/
├── main.py                     # 流水线入口
├── pyproject.toml              # 依赖声明（pandas / matplotlib / scipy；dev: pytest）
├── REPORT.md                   # 一次真实运行的完整分析报告（理论 + 结果）
├── TASK.md                     # 任务书（基于 NB/T 31147—2018）
├── data/                       # 输入 CSV（不入库）
│   ├── radar.csv
│   └── tower.csv
├── src/
│   ├── data_file.py            # CSV 导入/导出、时间解析
│   ├── radar/
│   │   ├── qc_filter.py        # 雷达 QC 流水线（40–200 m @ 5 m，33 层）
│   │   └── qc_cat.py           # 雷达可用率统计
│   ├── tower/
│   │   ├── qc_filter.py        # 测风塔 QC 流水线（2/5/10/20/50/80 m）
│   │   └── qc_cat.py           # 测风塔可用率统计
│   └── chart/
│       ├── weibull_plot.py     # Weibull MLE 拟合与绘图
│       ├── daily_variation.py  # 风速日变化
│       ├── density_variation.py# 空气密度日变化
│       ├── shear_variation.py  # 风切变指数日变化
│       └── wind_rose.py        # 风向/风能玫瑰图
├── test/
│   ├── test_data_file.py
│   ├── test_radar_qc.py
│   └── test_tower_qc.py
└── result/                     # 输出目录（不入库）
    ├── qc_radar.csv / qc_tower.csv
    └── chart/{radar,tower}/{weibull,daily_variation,density_variation,shear_variation,wind_rose}/
```

## 数据格式约定

### 雷达数据（SODAR）

- 测量高度：40–200 m，每 5 m 一层（共 33 层）
- 每层列名：`Wind Speed{h}m`、`Wind Direction{h}m`、
  `Vertical Wind Speed{h}m`、`Wind Speed Std{h}m`
- 站级参数：`Temperature`、`Pressure`、`Humidity`、`Battery Voltage`
- 时间列：`Time`

### 测风塔数据

- 测量高度：2 / 5 / 10 / 20 / 50 / 80 m
- 每层列名：`Avg Wind Speed @ {h}m [m/s]`、`Avg Wind Direction @ {h}m [deg]`、
  `Avg Wind Speed (std dev) @ {h}m [m/s]`、`Avg Wind Direction (std dev) @ {h}m [deg]`
- 站级参数：`Temperature @ {2,50,80}m [deg C]`、`Station Pressure [mBar]`、
  `Relative Humidity [%]`
- 时间列：`timestamp`

## QC 设计说明

- **执行顺序**：范围 → 缺测 → 相关性 → 趋势 → （廓线，禁用）→ 统一清洗。
  由简单到复杂、由局部到整体；缺测值先隔离，避免 NaN 进入层间差分。
- **编码优先级**：记录保留*首个*命中检验的编码（编码只升不覆盖），
  因此分步追踪（缺测优先）与主流程（范围优先）的 `QC > 0` 集合完全一致。
- **阈值取舍**：标准给出的是参考范围（如相邻层风速差 ≈ 2–5 m/s）；本工具
  对层间风速差采用更宽松的 15 m/s——33 层密集雷达数据在高原复杂地形下
  层间梯度可能显著偏大，过严会误杀大量合理数据。60 min 平线窗口比标准
  的 6 h 更严格，可更早发现传感器冻结。

## 依赖

| 库 | 最低版本 | 用途 |
|---|---|---|
| pandas | 3.0.3 | 数据处理、CSV 读写、滚动窗口 |
| matplotlib | 3.11.0 | 图表（子图、极坐标、直方图） |
| scipy | 1.17.1 | Weibull MLE 拟合 |
| pytest | 9.1.0 | 单元测试（开发依赖） |

## 许可证

暂未声明开源许可证，如需使用请联系作者。
