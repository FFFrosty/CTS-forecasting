"""数据规律量化分析 + 可视化：CTS2026 港口拖轮 AIS 数据。

输出 7 个分析模块的表格化结论 + 4 张规律图：
  1. 数据加载
  2. A题时序规律量化（含图）
  3. B题规律量化
  4. A↔B 相关性
  5. 单船规律量化
  6. 异常期量化
  7. 关键发现摘要

分析函数内部按需补齐完整时间网格；主流程不再做全局补零。
短表格直接控制台打印；较宽的矩阵（3×24、6×24、6×7、6×3）及图片存到
data/processed/explore/ 下。

运行：
    python scripts/explore.py


======================================================================
  Task 1: 数据加载
======================================================================

======================================================================
  Task 2: A题时序规律量化
======================================================================

----- 表1：每日总活跃拖轮·小时数（24天） -----
      date day_of_week_name  vessel_count
2018-01-01             周一           328
2018-01-02             周二           370
2018-01-03             周三           365
2018-01-04             周四           313
2018-01-05             周五           319
2018-01-06             周六           294
2018-01-07             周日           361
2018-01-08             周一           301
2018-01-09             周二           296
2018-01-10             周三           260
2018-01-11             周四           264
2018-01-12             周五           222
2018-01-19             周五           247
2018-01-20             周六           400
2018-01-21             周日           403
2018-01-22             周一           261
2018-01-23             周二           321
2018-01-24             周三           356

----- 表2：按星期几聚合（均值/std/CV/样本数） -----
day_name       mean       std       cv  n_samples
    周一 296.666667 33.709544 0.113628          3
    周二 329.000000 37.643060 0.114417          3
    周三 327.000000 58.197938 0.177975          3
    周四 288.500000 34.648232 0.120098          2
    周五 262.666667 50.362023 0.191734          3
    周六 347.000000 74.953319 0.216004          2
    周日 382.000000 29.698485 0.077745          2

----- 表3a：每圈层×每小时均值（行=圈层, 列=小时） -----
hour      0     1     2     3     4     5     6     7     8     9     10    11    12    13    14    15    16    17    18     19    20    21    22    23
zone                                                                                                                                                   
核心区  6.33  2.39  5.39  9.22  6.67  2.83  7.67  8.67  6.22  3.83  7.89  9.78  7.33  5.22  7.06  9.50  7.33  4.89  7.89  12.11  7.67  3.67  6.56  9.67
近港区  3.22  1.11  2.00  4.44  2.50  1.50  3.39  5.56  5.00  3.61  5.67  7.22  4.89  4.00  5.22  7.06  4.89  3.00  5.44   7.89  5.28  2.17  3.39  5.39
外围区  1.56  1.83  0.83  1.06  1.44  1.72  1.00  1.94  2.39  2.94  2.44  2.28  3.56  3.28  2.44  2.06  2.61  1.94  1.61   0.89  1.94  2.22  1.06  0.94

----- 表3b：每圈层×每小时标准差 -----
hour      0     1     2     3     4     5     6     7     8     9     10    11    12    13    14    15    16    17    18    19    20    21    22    23
zone                                                                                                                                                  
核心区  3.88  2.75  4.22  4.17  3.97  2.90  4.89  4.14  3.14  2.41  3.41  3.08  2.89  3.99  3.37  3.97  4.19  3.20  2.59  3.16  3.61  3.01  3.65  3.60
近港区  2.51  1.88  1.61  2.81  2.46  1.79  1.91  3.35  2.54  2.33  2.17  3.32  2.74  2.35  2.49  3.42  2.78  2.77  3.33  2.87  3.03  2.04  2.48  1.91
外围区  1.69  1.58  1.29  1.16  1.10  1.13  0.77  1.86  1.46  1.86  1.54  1.41  1.85  1.64  1.42  1.70  2.12  1.35  1.46  0.96  0.73  1.17  0.73  0.94

----- 表4：同星期同时刻跨周CV分布统计 -----
        min       25%       50%       75%       max      mean
跨周CV  0.0  0.204124  0.388145  0.707107  1.414214  0.478347
  （CV样本数=504，CV中位数=0.388）

----- 表4b：各圈层跨周CV中位数 -----
        cv_median
zone             
核心区   0.340756
近港区   0.367103
外围区   0.500000

======================================================================
  Task 3: B题规律量化
======================================================================

----- 表1：6方向稀疏度（非零比例/均值/最大值） -----
source_zone target_zone  total_samples  nonzero_count  nonzero_ratio     mean  max
     核心区      近港区            432            208       0.481481 1.034722    9
     核心区      外围区            432             77       0.178241 0.226852    3
     近港区      核心区            432            236       0.546296 1.081019    9
     近港区      外围区            432            102       0.236111 0.326389    5
     外围区      核心区            432             63       0.145833 0.175926    3
     外围区      近港区            432            116       0.268519 0.381944    5

----- 表2：每方向按星期几均值（6×7） -----
                 周一   周二   周三   周四   周五   周六   周日
dir                                                            
核心区->近港区  0.903  0.958  1.361  1.104  0.944  0.896  1.062
核心区->外围区  0.194  0.333  0.208  0.208  0.194  0.208  0.229
近港区->核心区  0.889  1.097  1.319  1.125  1.056  0.792  1.271
近港区->外围区  0.222  0.417  0.250  0.375  0.278  0.458  0.354
外围区->核心区  0.167  0.306  0.111  0.188  0.125  0.188  0.146
外围区->近港区  0.319  0.431  0.264  0.458  0.347  0.562  0.375

----- 表3：每方向24小时均值模式（6×24） -----
hour               0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23
dir                                                                                                                                                                                   
核心区->近港区  0.000  0.111  1.333  0.833  0.611  0.389  2.333  1.333  0.611  0.944  2.056  1.222  0.500  0.667  2.000  1.500  0.889  0.500  2.389  1.056  0.278  0.500  1.611  1.167
核心区->外围区  0.056  0.000  0.056  0.389  0.056  0.167  1.000  0.667  0.389  0.000  0.278  0.500  0.278  0.222  0.111  0.278  0.056  0.056  0.056  0.500  0.056  0.000  0.111  0.167
近港区->核心区  1.222  0.167  0.611  1.222  0.611  0.556  0.500  1.722  1.389  0.778  1.222  1.611  1.222  1.056  0.889  1.778  1.611  0.667  1.167  1.889  1.722  0.444  0.667  1.222
近港区->外围区  0.278  0.222  0.000  0.389  0.167  0.056  0.167  0.556  0.611  0.389  0.278  0.833  0.611  0.167  0.278  0.444  0.111  0.444  0.056  1.000  0.278  0.000  0.000  0.500
外围区->核心区  0.000  0.167  0.333  0.111  0.000  0.167  0.278  0.056  0.167  0.278  0.444  0.222  0.222  0.222  0.111  0.000  0.278  0.056  0.389  0.167  0.000  0.167  0.222  0.167
外围区->近港区  0.000  0.444  0.111  0.056  0.278  0.944  0.056  1.000  0.222  0.667  0.667  0.222  0.333  1.000  0.444  0.556  0.333  0.556  0.722  0.056  0.111  0.222  0.111  0.056

======================================================================
  Task 4: A-B 相关性
======================================================================

----- 6方向×3圈层 Pearson 相关矩阵 -----
                核心区  近港区  外围区
核心区->近港区   0.298   0.175  -0.157
核心区->外围区   0.240   0.197  -0.029
近港区->核心区   0.316   0.465   0.022
近港区->外围区   0.231   0.313   0.047
外围区->核心区  -0.062  -0.019   0.195
外围区->近港区  -0.122   0.024   0.422

  （对齐时间窗口数=432）

======================================================================
  Task 5: 单船规律量化
======================================================================

----- 表1：出勤天数分布（每船24天内出现的不同日期数） -----
 出勤天数  船数
        1     7
        2    11
        3     7
        4     6
        5     5
        6     5
        7     4
        8     2
        9     3
       10     1
       11     1
       12     4
       13     3
       16     1
       17     1
       18    31
  总船数=92，出勤天数中位数=8，均值=9.8

----- 表2：同星期同时刻跨周复现率分布 -----
复现率区间  样本数
     0-25%    6548
    25-50%     737
    50-75%    1394
   75-100%     955
  （总(船×星期×小时)组合数=9634，复现率中位数=0.00）

----- 表3：个体稳定性（每船每日活跃小时数的CV分布） -----
 CV区间  船数
  0-25%    20
 25-50%    30
50-100%    13
  >100%     2
  （有活跃记录的船数=78，CV中位数=0.34）

======================================================================
  Task 6: 异常期量化
======================================================================

----- 表1：异常期各圈层活跃量 z-score（3圈层×3天） -----
  注：周末 normal 仅1天样本，std 不可用时回退到 (zone,hour) 池化统计
      date   zone  mean_zscore  max_zscore  min_zscore  actual_sum  normal_mean_sum
2018-01-13 核心区          NaN         NaN         NaN           0              0.0
2018-01-13 近港区          NaN         NaN         NaN           0              0.0
2018-01-13 外围区          NaN         NaN         NaN           0              0.0
2018-01-14 核心区          NaN         NaN         NaN           0              0.0
2018-01-14 近港区          NaN         NaN         NaN           0              0.0
2018-01-14 外围区          NaN         NaN         NaN           0              0.0
2018-01-15 核心区          NaN         NaN         NaN           0              0.0
2018-01-15 近港区          NaN         NaN         NaN           0              0.0
2018-01-15 外围区          NaN         NaN         NaN           0              0.0

----- 表2：异常期每日总活跃量 vs normal 均值 -----
  normal 期每日总量均值: 315.5
  normal 期每日总量std: 38.0
  2018-01-13: 总量=0, z-score=-8.30
  2018-01-14: 总量=0, z-score=-8.30
  2018-01-15: 总量=0, z-score=-8.30

######################################################################
  ===== 关键发现摘要 =====
######################################################################

1. 周周期稳定性：同星期同时刻跨周CV中位数 = 0.388（各圈层：核心区=0.341  近港区=0.367  外围区=0.500  ）。周周期较稳定。

2. B题稀疏度：6方向非零比例范围为 14.6% ~ 54.6%。
   无方向非零比例低于5%。

3. A-B 相关性：6×3相关矩阵范围 [-0.16, 0.47]。相关性较弱，B题难以仅靠A题活跃量推算。

4. 单船出勤：总船数=92，出勤天数中位数=8，出勤≥20天的稳定船数=0（0.0%）。逐船建模样本不足。

5. 异常期（1/13-1/15）z-score：各圈层日均z-score范围 [nan, nan]，最极端单点z-score=nan。异常期偏离幅度有限。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==================== 全局设置 ====================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.unicode.east_asian_width", True)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 项目根目录（不依赖 sys.path hack）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPLOR_DIR = PROCESSED_DIR / "explore"

# ==================== 常量 ====================
ZONES = ["核心区", "近港区", "外围区"]
ZONE_BIT = {"外围区": 1, "近港区": 2, "核心区": 4}
DIRECTIONS = [
    ("核心区", "近港区"), ("核心区", "外围区"),
    ("近港区", "核心区"), ("近港区", "外围区"),
    ("外围区", "核心区"), ("外围区", "近港区"),
]
TRAIN_START = "2018-01-01"
TRAIN_END = "2018-01-24"  # 含当天，到 2018-01-24 23:00
ALL_HOURS = pd.date_range(f"{TRAIN_START} 00:00", f"{TRAIN_END} 23:00", freq="h")

DOW_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# ==================== Task 1: 数据加载 ====================
def load_csv_or_exit(filename: str) -> pd.DataFrame:
    """加载 processed CSV，缺失则报错退出。"""
    path = PROCESSED_DIR / filename
    df = pd.read_csv(path, parse_dates=["time_window"])
    return df



def add_time_info(df: pd.DataFrame) -> pd.DataFrame:
    """补齐 hour / day_of_week / date 列。"""
    df = df.copy()
    df["hour"] = df["time_window"].dt.hour
    df["day_of_week"] = df["time_window"].dt.dayofweek  # 0=周一
    df["date"] = df["time_window"].dt.date
    return df


def print_section(title: str) -> None:
    """打印带分隔线的章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ==================== Task 2: A题时序规律量化 ====================
def task2_a_stats(a_df: pd.DataFrame) -> dict:
    """A 题时序规律量化，返回 4 张表。"""
    print_section("Task 2: A题时序规律量化")

    # 表1：每日总活跃拖轮·小时数
    daily = (
        a_df.groupby("date")["vessel_count"].sum().reset_index()
    )
    daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.dayofweek
    daily["day_of_week_name"] = daily["day_of_week"].map(lambda i: DOW_NAMES[i])
    daily = daily[["date", "day_of_week_name", "vessel_count"]]
    print("\n----- 表1：每日总活跃拖轮·小时数（24天） -----")
    print(daily.to_string(index=False))

    # 表2：按星期几聚合
    dow_agg = (
        a_df.groupby(["date", "day_of_week"])["vessel_count"].sum()
        .reset_index()
        .groupby("day_of_week")["vessel_count"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    dow_agg["cv"] = dow_agg["std"] / dow_agg["mean"]
    dow_agg["day_name"] = dow_agg["day_of_week"].map(lambda i: DOW_NAMES[i])
    dow_agg = dow_agg[["day_name", "mean", "std", "cv", "count"]]
    dow_agg.columns = ["day_name", "mean", "std", "cv", "n_samples"]
    print("\n----- 表2：按星期几聚合（均值/std/CV/样本数） -----")
    print(dow_agg.to_string(index=False))



    # 表3：每圈层×每小时均值±std
    zone_hour = (
        a_df.groupby(["zone", "hour"])["vessel_count"]
        .agg(["mean", "std"])
        .reset_index()
    )
    # 转成 3×24 矩阵（行=圈层，列=小时）
    mean_mat = zone_hour.pivot(index="zone", columns="hour", values="mean")
    std_mat = zone_hour.pivot(index="zone", columns="hour", values="std")
    mean_mat = mean_mat.reindex(ZONES)
    std_mat = std_mat.reindex(ZONES)
    print("\n----- 表3a：每圈层×每小时均值（行=圈层, 列=小时） -----")
    print(mean_mat.round(2).to_string())
    print("\n----- 表3b：每圈层×每小时标准差 -----")
    print(std_mat.round(2).to_string())

    # 表4：同星期同时刻跨周CV分布
    a_df["week_idx"] = (
        (pd.to_datetime(a_df["date"]) - pd.Timestamp(TRAIN_START)).dt.days // 7
    )
    grouped = a_df.groupby(["zone", "day_of_week", "hour", "week_idx"])[
        "vessel_count"
    ].sum()
    # 每个 (zone, dow, hour) 跨周的 CV
    cv_records = []
    for (zone, dow, hour), sub in grouped.groupby(level=["zone", "day_of_week", "hour"]):
        vals = sub.values
        if len(vals) < 2:
            continue
        m = vals.mean()
        if m == 0:
            cv = 0.0
        else:
            cv = vals.std(ddof=0) / m
        cv_records.append({"zone": zone, "day_of_week": dow, "hour": hour, "cv": cv})
    cv_df = pd.DataFrame(cv_records)
    # 输出 CV 的分布统计
    cv_dist = cv_df["cv"].describe(
        percentiles=[0.25, 0.5, 0.75]
    )[["min", "25%", "50%", "75%", "max", "mean"]]
    cv_dist_df = cv_dist.to_frame(name="跨周CV").T
    print("\n----- 表4：同星期同时刻跨周CV分布统计 -----")
    print(cv_dist_df.to_string())
    print(f"  （CV样本数={len(cv_df)}，CV中位数={cv_df['cv'].median():.3f}）")

    # 按圈层细分 CV 中位数
    cv_by_zone = cv_df.groupby("zone")["cv"].median().reindex(ZONES)
    print("\n----- 表4b：各圈层跨周CV中位数 -----")
    print(cv_by_zone.to_frame("cv_median").to_string())

    return {
        "daily": daily,
        "dow_agg": dow_agg,
        "mean_mat": mean_mat,
        "std_mat": std_mat,
        "cv_dist": cv_dist_df,
        "cv_by_zone": cv_by_zone,
        "cv_df": cv_df,
    }


# ==================== Task 3: B题规律量化 ====================
def task3_b_stats(b_df: pd.DataFrame) -> dict:
    """B 题规律量化。"""
    print_section("Task 3: B题规律量化")

    b_df["dir"] = b_df["source_zone"] + "->" + b_df["target_zone"]
    dir_order = [f"{s}->{t}" for s, t in DIRECTIONS]

    # 表1：6方向稀疏度
    sparse_rows = []
    for src, tgt in DIRECTIONS:
        sub = b_df[(b_df["source_zone"] == src) & (b_df["target_zone"] == tgt)]
        vc = sub["vessel_count"]
        nonzero = (vc > 0).sum()
        total = len(vc)
        sparse_rows.append({
            "source_zone": src,
            "target_zone": tgt,
            "total_samples": total,
            "nonzero_count": int(nonzero),
            "nonzero_ratio": nonzero / total if total else 0,
            "mean": vc.mean(),
            "max": vc.max(),
        })
    sparse_df = pd.DataFrame(sparse_rows)
    print("\n----- 表1：6方向稀疏度（非零比例/均值/最大值） -----")
    print(sparse_df.to_string(index=False))

    # 表2：每方向按星期几均值（6×7）
    dow_mean = b_df.groupby(["dir", "day_of_week"])["vessel_count"].mean().unstack()
    dow_mean = dow_mean.reindex(dir_order)
    dow_mean.columns = DOW_NAMES
    print("\n----- 表2：每方向按星期几均值（6×7） -----")
    print(dow_mean.round(3).to_string())

    # 表3：每方向24小时模式（6×24）
    hour_mean = b_df.groupby(["dir", "hour"])["vessel_count"].mean().unstack()
    hour_mean = hour_mean.reindex(dir_order)
    print("\n----- 表3：每方向24小时均值模式（6×24） -----")
    print(hour_mean.round(3).to_string())

    return {
        "sparse": sparse_df,
        "dow_mean": dow_mean,
        "hour_mean": hour_mean,
    }


# ==================== Task 4: A↔B 相关性 ====================
def task4_ab_corr(a_df: pd.DataFrame, b_df: pd.DataFrame) -> pd.DataFrame:
    """A↔B 相关性：6×3 Pearson 相关矩阵。"""
    print_section("Task 4: A-B 相关性")

    b_df["dir"] = b_df["source_zone"] + "->" + b_df["target_zone"]
    dir_order = [f"{s}->{t}" for s, t in DIRECTIONS]

    # A 题：每个时间窗口 3 圈层活跃量
    a_pivot = a_df.pivot_table(
        index="time_window", columns="zone", values="vessel_count", fill_value=0
    )
    a_pivot = a_pivot.reindex(columns=ZONES)

    # B 题：每个时间窗口 6 方向流量
    b_pivot = b_df.pivot_table(
        index="time_window", columns="dir", values="vessel_count", fill_value=0
    )
    b_pivot = b_pivot.reindex(columns=dir_order)

    # 对齐时间窗口
    common_idx = a_pivot.index.intersection(b_pivot.index)
    a_aligned = a_pivot.loc[common_idx]
    b_aligned = b_pivot.loc[common_idx]

    # 6×3 相关矩阵
    corr_mat = pd.DataFrame(index=dir_order, columns=ZONES, dtype=float)
    for d in dir_order:
        for z in ZONES:
            if a_aligned[z].std() == 0 or b_aligned[d].std() == 0:
                corr_mat.loc[d, z] = np.nan
            else:
                corr_mat.loc[d, z] = a_aligned[z].corr(b_aligned[d])

    print("\n----- 6方向×3圈层 Pearson 相关矩阵 -----")
    print(corr_mat.round(3).to_string())
    print(f"\n  （对齐时间窗口数={len(common_idx)}）")

    return corr_mat


# ==================== Task 5: 单船规律量化 ====================
def task5_vessel_stats(vessel_state: pd.DataFrame) -> dict:
    """单船规律量化（基于 vessel_state.csv）。"""
    print_section("Task 5: 单船规律量化")

    vs = vessel_state.copy()
    vs["hour"] = vs["time_window"].dt.hour
    vs["day_of_week"] = vs["time_window"].dt.dayofweek
    vs["date"] = vs["time_window"].dt.date
    # 出勤定义：is_active=True 或 zone_state>0
    vs["present"] = vs["is_active"] | (vs["zone_state"] > 0)

    # 表1：出勤天数分布
    attend_days = vs.groupby("mmsi")["date"].nunique()
    attend_dist = attend_days.value_counts().sort_index().reset_index()
    attend_dist.columns = ["出勤天数", "船数"]
    print("\n----- 表1：出勤天数分布（每船24天内出现的不同日期数） -----")
    print(attend_dist.to_string(index=False))
    print(f"  总船数={len(attend_days)}，出勤天数中位数={attend_days.median():.0f}，"
          f"均值={attend_days.mean():.1f}")

    # 表2：同星期同时刻复现率
    # 对每船每个 (dow, hour)，统计活跃周数 / 总周数
    vs["week_idx"] = (
        (pd.to_datetime(vs["date"]) - pd.Timestamp(TRAIN_START)).dt.days // 7
    )
    # 每个 (mmsi, dow, hour, week_idx) 该周是否活跃
    week_active = (
        vs.groupby(["mmsi", "day_of_week", "hour", "week_idx"])["present"]
        .any()
        .reset_index()
    )
    # 每个 (mmsi, dow, hour) 的活跃周数 / 总周数
    repro = (
        week_active.groupby(["mmsi", "day_of_week", "hour"])["present"]
        .agg(["sum", "count"])
        .reset_index()
    )
    repro["rate"] = repro["sum"] / repro["count"]
    # 分桶
    bins = [0, 0.25, 0.5, 0.75, 1.0 + 1e-9]
    labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    repro["bucket"] = pd.cut(repro["rate"], bins=bins, labels=labels, right=False)
    repro_dist = repro["bucket"].value_counts().reindex(labels, fill_value=0)
    repro_dist_df = repro_dist.reset_index()
    repro_dist_df.columns = ["复现率区间", "样本数"]
    print("\n----- 表2：同星期同时刻跨周复现率分布 -----")
    print(repro_dist_df.to_string(index=False))
    print(f"  （总(船×星期×小时)组合数={len(repro)}，"
          f"复现率中位数={repro['rate'].median():.2f}）")

    # 表3：个体稳定性——每船每日活跃小时数
    daily_active = (
        vs[vs["is_active"]].groupby(["mmsi", "date"]).size().reset_index(name="active_hours")
    )
    vessel_stab = (
        daily_active.groupby("mmsi")["active_hours"]
        .agg(["mean", "std"])
        .reset_index()
    )
    vessel_stab["cv"] = vessel_stab["std"] / vessel_stab["mean"]
    cv_bins = [0, 0.25, 0.5, 1.0, np.inf]
    cv_labels = ["0-25%", "25-50%", "50-100%", ">100%"]
    vessel_stab["cv_bucket"] = pd.cut(
        vessel_stab["cv"], bins=cv_bins, labels=cv_labels, right=False
    )
    stab_dist = vessel_stab["cv_bucket"].value_counts().reindex(cv_labels, fill_value=0)
    stab_dist_df = stab_dist.reset_index()
    stab_dist_df.columns = ["CV区间", "船数"]
    print("\n----- 表3：个体稳定性（每船每日活跃小时数的CV分布） -----")
    print(stab_dist_df.to_string(index=False))
    print(f"  （有活跃记录的船数={len(vessel_stab)}，"
          f"CV中位数={vessel_stab['cv'].median():.2f}）")

    return {
        "attend_dist": attend_dist,
        "repro_dist": repro_dist_df,
        "stab_dist": stab_dist_df,
        "attend_days": attend_days,
        "repro": repro,
        "vessel_stab": vessel_stab,
    }


# ==================== Task 6: 异常期量化 ====================
def task6_anomaly(a_df: pd.DataFrame) -> dict:
    """异常期量化：normal=1/1-1/11, anomaly=1/13-1/15。"""
    print_section("Task 6: 异常期量化")

    normal_mask = (a_df["date"] >= pd.Timestamp("2018-01-01").date()) & \
                  (a_df["date"] <= pd.Timestamp("2018-01-11").date())
    anomaly_dates = [pd.Timestamp(f"2018-01-{d}").date() for d in (13, 14, 15)]

    # normal 期的 (zone, dow, hour) 统计
    normal_df = a_df[normal_mask]
    normal_stats = normal_df.groupby(["zone", "day_of_week", "hour"])[
        "vessel_count"
    ].agg(["mean", "std"]).reset_index()
    # 回退统计：normal 期 (zone, hour) 池化统计（用于 dow 样本不足时，如周末仅1天）
    normal_stats_pool = normal_df.groupby(["zone", "hour"])[
        "vessel_count"
    ].agg(["mean", "std"]).reset_index().rename(
        columns={"mean": "mean_pool", "std": "std_pool"}
    )

    # 表1：各圈层活跃量 z-score（3圈层×3天=9行）
    # 优先用 (zone, dow, hour) 的 std；若 NaN/0（样本不足），回退到 (zone, hour) 池化统计
    zscore_rows = []
    for ad in anomaly_dates:
        ad_df = a_df[a_df["date"] == ad]
        for zone in ZONES:
            zdf = ad_df[ad_df["zone"] == zone].merge(
                normal_stats, on=["zone", "day_of_week", "hour"], how="left"
            ).merge(
                normal_stats_pool, on=["zone", "hour"], how="left"
            )
            # dow 特定 std 不可用时回退到池化统计
            need_fallback = zdf["std"].isna() | (zdf["std"] == 0)
            zdf.loc[need_fallback, "mean"] = zdf.loc[need_fallback, "mean_pool"]
            zdf.loc[need_fallback, "std"] = zdf.loc[need_fallback, "std_pool"]
            std = zdf["std"].replace(0, np.nan)
            z = (zdf["vessel_count"] - zdf["mean"]) / std
            zscore_rows.append({
                "date": str(ad),
                "zone": zone,
                "mean_zscore": z.mean(),
                "max_zscore": z.max(),
                "min_zscore": z.min(),
                "actual_sum": zdf["vessel_count"].sum(),
                "normal_mean_sum": zdf["mean"].sum(),
            })
    zscore_df = pd.DataFrame(zscore_rows)
    print("\n----- 表1：异常期各圈层活跃量 z-score（3圈层×3天） -----")
    print("  注：周末 normal 仅1天样本，std 不可用时回退到 (zone,hour) 池化统计")
    print(zscore_df.round(3).to_string(index=False))

    # 表2：异常期每日总活跃量对比 normal 均值
    normal_daily = normal_df.groupby("date")["vessel_count"].sum()
    print("\n----- 表2：异常期每日总活跃量 vs normal 均值 -----")
    print(f"  normal 期每日总量均值: {normal_daily.mean():.1f}")
    print(f"  normal 期每日总量std: {normal_daily.std():.1f}")
    for ad in anomaly_dates:
        ad_sum = a_df[a_df["date"] == ad]["vessel_count"].sum()
        z = (ad_sum - normal_daily.mean()) / normal_daily.std()
        print(f"  {ad}: 总量={ad_sum:.0f}, z-score={z:.2f}")

    return {"zscore": zscore_df, "normal_daily_mean": normal_daily.mean()}


# ==================== Task 7: 关键发现摘要 ====================
def task7_summary(
    a_stats: dict, b_stats: dict, corr_mat: pd.DataFrame,
    vessel_stats: dict, anomaly: dict
) -> None:
    """打印关键发现摘要。"""
    print("\n" + "#" * 70)
    print("  ===== 关键发现摘要 =====")
    print("#" * 70)

    # 1. 周周期稳定性
    cv_median = a_stats["cv_df"]["cv"].median()
    print(f"\n1. 周周期稳定性：同星期同时刻跨周CV中位数 = {cv_median:.3f}（各圈层：", end="")
    for z in ZONES:
        print(f"{z}={a_stats['cv_by_zone'][z]:.3f}", end="  ")
    print(("）。周周期较稳定" if cv_median < 0.5 else "）。周周期波动较大") + "。")

    # 2. B题稀疏度
    sparse = b_stats["sparse"]
    low_ratio = sparse[sparse["nonzero_ratio"] < 0.05]
    print(f"\n2. B题稀疏度：6方向非零比例范围为 "
          f"{sparse['nonzero_ratio'].min():.1%} ~ {sparse['nonzero_ratio'].max():.1%}。")
    if len(low_ratio) > 0:
        dirs = [f"{r['source_zone']}->{r['target_zone']}" for _, r in low_ratio.iterrows()]
        print(f"   非零比例极低（<5%）的方向: {', '.join(dirs)}，可考虑直接填0。")
    else:
        print("   无方向非零比例低于5%。")

    # 3. A↔B 相关性
    corr_max = corr_mat.max().max()
    corr_min = corr_mat.min().min()
    print(f"\n3. A-B 相关性：6×3相关矩阵范围 [{corr_min:.2f}, {corr_max:.2f}]。"
          f"{'相关性较强' if abs(corr_max) > 0.5 or abs(corr_min) > 0.5 else '相关性较弱'}，"
          f"{'B题可用A题活跃量推算' if abs(corr_max) > 0.5 else 'B题难以仅靠A题活跃量推算'}。")

    # 4. 单船出勤
    attend = vessel_stats["attend_days"]
    stable_vessels = (attend >= 20).sum()
    print(f"\n4. 单船出勤：总船数={len(attend)}，"
          f"出勤天数中位数={attend.median():.0f}，"
          f"出勤≥20天的稳定船数={stable_vessels}（{stable_vessels/len(attend):.1%}）。"
          f"{'逐船建模可行' if stable_vessels > 10 else '逐船建模样本不足'}。")

    # 5. 异常期 z-score
    z = anomaly["zscore"]
    min_z = z["min_zscore"].min()
    print(f"\n5. 异常期（1/13-1/15）z-score："
          f"各圈层日均z-score范围 [{z['mean_zscore'].min():.2f}, {z['mean_zscore'].max():.2f}]，"
          f"最极端单点z-score={min_z:.2f}。"
          f"{'异常期偏离显著，需特殊处理' if z['mean_zscore'].min() < -1 else '异常期偏离幅度有限'}。")


# ==================== A题可视化 ====================
def plot_a_analysis(a_full: pd.DataFrame, output_dir: Path) -> None:
    """A 题 4 张规律图（含 AIS 条目数背景参考）。"""
    ZONES = ["核心区", "近港区", "外围区"]
    has_ais = "n_records" in a_full.columns

    # AIS 条目数背景数据（每小时一条，去重）
    if has_ais:
        ais_ts = (
            a_full[["time_window", "n_records"]]
            .drop_duplicates("time_window")
            .sort_values("time_window")
        )

    # --- 图1：全时序 ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    for i, z in enumerate(ZONES):
        d = a_full[a_full["zone"] == z]
        axes[i].plot(d["time_window"], d["vessel_count"], linewidth=0.6, alpha=0.7)
        if has_ais:
            ax2 = axes[i].twinx()
            ax2.fill_between(ais_ts["time_window"], 0, ais_ts["n_records"], alpha=0.06, color="gray")
            ax2.set_yticks([])
        axes[i].set_ylabel(z)
        axes[i].grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.suptitle("24天训练集——各圈层每小时活跃拖轮数（灰色背景=AIS条目数）", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_full_timeline.png", dpi=120)
    plt.close()

    # --- 图2：每周叠加 ---
    a_full["week_label"] = (
        "week" + ((pd.to_datetime(a_full["date"]) - pd.Timestamp("2018-01-01")).dt.days // 7 + 1).astype(str)
    )
    a_full["hour_of_week"] = a_full["day_of_week"] * 24 + a_full["hour"]
    if has_ais:
        ais_wow = a_full.groupby("hour_of_week")["n_records"].mean().reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    for i, z in enumerate(ZONES):
        d = a_full[a_full["zone"] == z]
        if has_ais:
            ax2 = axes[i].twinx()
            ax2.fill_between(ais_wow["hour_of_week"], 0, ais_wow["n_records"], alpha=0.06, color="gray")
            ax2.set_yticks([])
        for wl in d["week_label"].unique():
            w = d[d["week_label"] == wl]
            axes[i].plot(w["hour_of_week"], w["vessel_count"], linewidth=0.7, alpha=0.6, label=wl)
        axes[i].set_ylabel(z)
        axes[i].legend(fontsize=7, loc="upper right")
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("hour of week (0=Mon 00:00)")
    fig.suptitle("各周重叠对比——周模式稳定性（灰色背景=AIS条目数）", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_weekly_overlay.png", dpi=120)
    plt.close()

    # --- 图3：24小时×7天热力图 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, z in enumerate(ZONES):
        piv = a_full[a_full["zone"] == z].pivot_table(
            index="hour", columns="day_of_week", values="vessel_count", aggfunc="mean"
        )
        im = axes[i].imshow(piv.values, aspect="auto", cmap="YlOrRd", origin="lower",
                            extent=[-0.5, 6.5, -0.5, 23.5])
        axes[i].set_title(z, fontsize=11)
        axes[i].set_xlabel("day of week")
        axes[i].set_xticks(range(7))
        axes[i].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        axes[i].set_ylabel("hour")
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    fig.suptitle("小时×星期均值热力图 —— 什么时段最忙？", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_heatmap.png", dpi=120)
    plt.close()

    # --- 图4：每小时波动范围 ---
    if has_ais:
        ais_hr = a_full.groupby("hour")["n_records"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, z in enumerate(ZONES):
        d = a_full[a_full["zone"] == z]
        stats = d.groupby("hour")["vessel_count"].agg(["mean", "std", "min", "max"])
        axes[i].fill_between(stats.index, stats["min"], stats["max"], alpha=0.15, color="steelblue")
        axes[i].fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                             alpha=0.3, color="steelblue")
        axes[i].plot(stats.index, stats["mean"], color="steelblue", linewidth=2, label="mean")
        if has_ais:
            ax2 = axes[i].twinx()
            ax2.plot(ais_hr.index, ais_hr.values, color="gray", linewidth=0.8, alpha=0.5, ls="--", label="AIS records")
            ax2.set_ylabel("AIS records", fontsize=7, color="gray")
            ax2.tick_params(axis="y", labelsize=6, colors="gray")
            if i == 0:
                ax2.legend(fontsize=6, loc="upper left")
        axes[i].set_title(z)
        axes[i].set_xlabel("hour")
        axes[i].set_ylabel("vessel_count")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    fig.suptitle("各圈层24小时模式——均值±1σ+极值范围（虚线=AIS条目数）", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_hourly_pattern.png", dpi=120)
    plt.close()


# ==================== 主流程 ====================
def main() -> None:
    print_section("Task 1: 数据加载")

    # 加载
    task_a = load_csv_or_exit("task_a_train.csv")
    task_b = load_csv_or_exit("task_b_train.csv")
    vessel_state = load_csv_or_exit("vessel_state.csv")

    a_df = add_time_info(task_a)
    b_df = add_time_info(task_b)

    # 合并 AIS 条目数（标注数据断档用）
    record_counts = load_csv_or_exit("ais_record_counts.csv")
    a_df = a_df.merge(record_counts, on="time_window", how="left")

    # 排除数据源断层期（1/13-1/18）
    anom_start = pd.Timestamp("2018-01-13").date()
    anom_end = pd.Timestamp("2018-01-18").date()
    a_df = a_df[(a_df["date"] < anom_start) | (a_df["date"] > anom_end)].copy()
    b_df = b_df[(b_df["date"] < anom_start) | (b_df["date"] > anom_end)].copy()
    vessel_state = vessel_state[
        (vessel_state["time_window"].dt.date < anom_start)
        | (vessel_state["time_window"].dt.date > anom_end)
    ].copy()

    EXPLOR_DIR.mkdir(parents=True, exist_ok=True)

    # Task 2
    a_stats = task2_a_stats(a_df)

    # Task 3
    b_stats = task3_b_stats(b_df)

    # Task 4
    corr_mat = task4_ab_corr(a_df, b_df)

    # Task 5
    vessel_stats = task5_vessel_stats(vessel_state)

    # Task 6
    anomaly = task6_anomaly(a_df)

    # Task 7
    task7_summary(a_stats, b_stats, corr_mat, vessel_stats, anomaly)

    # 画图
    plot_a_analysis(a_df, EXPLOR_DIR)

    # 只把稍微长、看文字费劲的矩阵存 CSV；短表格直接看控制台输出
    saved = []
    a_stats["mean_mat"].to_csv(EXPLOR_DIR / "a_zone_hour_mean.csv", encoding="utf-8-sig")
    saved.append("a_zone_hour_mean.csv")
    a_stats["std_mat"].to_csv(EXPLOR_DIR / "a_zone_hour_std.csv", encoding="utf-8-sig")
    saved.append("a_zone_hour_std.csv")
    b_stats["dow_mean"].to_csv(EXPLOR_DIR / "b_dow_mean.csv", encoding="utf-8-sig")
    saved.append("b_dow_mean.csv")
    b_stats["hour_mean"].to_csv(EXPLOR_DIR / "b_hour_mean.csv", encoding="utf-8-sig")
    saved.append("b_hour_mean.csv")
    corr_mat.to_csv(EXPLOR_DIR / "ab_corr.csv", encoding="utf-8-sig")
    saved.append("ab_corr.csv")


if __name__ == "__main__":
    main()
