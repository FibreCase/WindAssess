# WindAssess — Wind Resource Assessment Toolkit

English | [中文](README_zh.md)

> [Final Report (in Chinese)](REPORT.md)

A Python toolkit for processing and analyzing **wind resource measurement
data** from two complementary observation platforms — **SODAR / acoustic
radar** and **conventional met masts** — following the QC framework of
**NB/T 31147—2018** (*Technical specifications for wind energy resource
measurement and assessment of wind power plant projects*).

Built for wind-farm siting: multi-stage automated quality control,
effective data-completeness statistics, and the full set of classical
wind-resource parameters (Weibull distribution, diurnal variation, air
density, wind shear, wind/energy roses).

```
data/{radar,tower}.csv ──▶ QC pipeline (4-stage) ──▶ result/qc_*.csv
                                            └──────▶ result/chart/{radar,tower}/ (5 chart families)
```

## Data sources

| | SODAR (radar) | Met mast (tower) |
|---|---|---|
| Region | Tibet plateau (complex terrain) | Conventional terrain (plain/hill) |
| Heights | 40–200 m, 5 m step — **33 levels** | 2 / 5 / 10 / 20 / 50 / 80 m — **6 levels** |
| Records | ~10,718 × 10-min | ~52,950 × 10-min |
| Extra params | vertical wind speed, single-station T/P/RH | T at 3 heights, station pressure, RH |

The two datasets differ in technique, terrain, and vertical coverage, so the
toolkit runs a **symmetric pipeline** for each and supports side-by-side
comparison (see `REPORT.md` for a full comparative analysis of real runs).

## Features

- **Multi-stage QC** with a per-height QC code (first violation wins):
  - **Range test** (code 1) — physically impossible values: wind speed outside
    `[0, 75] m/s`, direction outside `[0°, 360°)`, vertical wind speed outside
    `[-10, 10] m/s`, wind-speed σ outside `[0, 20] m/s`
  - **Missing-value test** (code 5) — NaN wind speed or direction; isolated
    *before* inter-level comparisons so NaN never propagates
  - **Correlation tests** (code 2) — turbulence intensity `TI = σ/V` outside
    `[0, 1]`; adjacent-level wind-speed difference > **15 m/s**; adjacent-level
    wind-direction difference > **120°** (with a wrap-aware angle operator,
    so 350° vs 10° = 20°, not 340°)
  - **Trend tests** (code 3) — *flatline*: 6 consecutive identical readings
    (60 min window, 3-decimal rounding); *spike*: consecutive change >
    **20 m/s**
  - **Profile-consistency test** (code 4) — implemented (counts sign changes of
    the inter-level speed profile; flags unphysical oscillation) but currently
    **disabled** in the main pipeline for performance reasons
- **No interpolation** — flagged samples are masked to NaN only; data are
  reported as measured, per standard practice for evaluation-period statistics
- **Effective data completeness** per NB/T 31147—2018 §5.2.10
  (`γ = (Rₑ − Rᵢ − R_w)/Rₑ`), plus a **step-wise availability tracker** that
  reports how much each QC stage removed at every height
- **Wind-resource parameters** (each as per-height subplots + combined chart,
  PNG @ 300 DPI):
  - **Weibull distribution** — two-parameter MLE fit (`scipy.stats.weibull_min`,
    `floc=0`) of wind-speed PDFs; shape `k` / scale `c` reported
  - **Diurnal variation** — hourly mean ± σ and min–max envelope of wind speed
  - **Air density** — ideal-gas law `ρ = P/(R·T)` from measured T and P
  - **Wind shear** — power-law exponent `α = ln(V₂/V₁)/ln(h₂/h₁)` between
    adjacent levels, with the 1/7 reference line (α = 0.143)
  - **Wind roses** — 16-sector (22.5°) directional frequency rose and
    wind-energy-density rose (`½ρV³`, standard ρ = 1.225 kg/m³)
- **Unit-tested** QC core (range / correlation / trend logic per source)

## Quick start

Requirements: **Python ≥ 3.13**, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/FibreCase/WindAssess.git
cd WindAssess
uv sync

# put your data files in data/
#   data/radar.csv   (SODAR)
#   data/tower.csv   (met mast)

uv run python main.py
```

`main.py` runs the whole pipeline for both sources:

1. import `data/radar.csv` and `data/tower.csv` (time columns auto-parsed)
2. QC + per-stage availability tracking for each
3. export cleaned data → `result/qc_radar.csv`, `result/qc_tower.csv`
   (+ `result/qc_*_rate.csv` step-wise availability tables)
4. generate all five chart families → `result/chart/{radar,tower}/`

> Paths are hardcoded in `main.py`; edit them if your files live elsewhere.

### Tests

```bash
uv run pytest
```

## Project layout

```
WindAssess/
├── main.py                     # pipeline entry point
├── pyproject.toml              # deps (pandas, matplotlib, scipy; pytest in dev)
├── REPORT.md                   # full worked analysis (theory + results) for one run
├── data/                       # input CSVs (not committed)
│   ├── radar.csv
│   └── tower.csv
├── src/
│   ├── data_file.py            # CSV import/export, time parsing
│   ├── radar/
│   │   ├── qc_filter.py        # radar QC: 40–200 m @ 5 m (33 levels)
│   │   └── qc_cat.py           # radar availability stats
│   ├── tower/
│   │   ├── qc_filter.py        # tower QC: 2/5/10/20/50/80 m
│   │   └── qc_cat.py           # tower availability stats
│   └── chart/
│       ├── weibull_plot.py     # Weibull MLE fit + plots
│       ├── daily_variation.py  # hourly wind-speed variation
│       ├── density_variation.py# air-density diurnal curve
│       ├── shear_variation.py  # shear-exponent diurnal curve
│       └── wind_rose.py        # direction & energy roses
├── test/
│   ├── test_data_file.py
│   ├── test_radar_qc.py
│   └── test_tower_qc.py
└── result/                     # outputs (not committed)
    ├── qc_radar.csv / qc_tower.csv
    └── chart/{radar,tower}/{weibull,daily_variation,density_variation,shear_variation,wind_rose}/
```

## Data format

### Radar (SODAR)

- Heights: 40–200 m, 5 m step (33 levels)
- Per level: `Wind Speed{h}m`, `Wind Direction{h}m`,
  `Vertical Wind Speed{h}m`, `Wind Speed Std{h}m`
- Station: `Temperature`, `Pressure`, `Humidity`, `Battery Voltage`
- Time column: `Time`

### Tower (met mast)

- Heights: 2 / 5 / 10 / 20 / 50 / 80 m
- Per level: `Avg Wind Speed @ {h}m [m/s]`, `Avg Wind Direction @ {h}m [deg]`,
  `Avg Wind Speed (std dev) @ {h}m [m/s]`, `Avg Wind Direction (std dev) @ {h}m [deg]`
- Station: `Temperature @ {2,50,80}m [deg C]`, `Station Pressure [mBar]`,
  `Relative Humidity [%]`
- Time column: `timestamp`

## QC design notes

- **Execution order**: range → missing → correlation → trend → (profile,
  disabled) → cleanup. Simple/local checks run before expensive/global ones;
  missing values are isolated before any inter-level difference is computed.
- **Code priority**: a record keeps the code of the *first* test that flagged
  it (codes are never overwritten), so `QC > 0` sets are identical regardless
  of the order used for the step-wise tracker.
- **Thresholds vs. the standard**: the standard gives reference ranges
  (e.g. adjacent-level ΔV ≈ 2–5 m/s); this toolkit uses a deliberately
  looser 15 m/s inter-level threshold because 33 closely spaced SODAR levels
  over plateau terrain produce large but legitimate gradients — the tighter
  standard limits would over-flag. The 60-min flatline window is stricter
  than the standard's 6 h to catch sensor freezes early.

## Dependencies

| Package | Min version | Role |
|---|---|---|
| pandas | 3.0.3 | data handling, CSV I/O, rolling windows |
| matplotlib | 3.11.0 | all charts (subplots, polar, histograms) |
| scipy | 1.17.1 | Weibull MLE fitting |
| pytest | 9.1.0 | unit tests (dev) |

## License

No license declared yet — see repository settings before reusing.
