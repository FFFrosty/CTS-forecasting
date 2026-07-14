"""探索更优预测周期：不限于星期，扫多种分组方式比较CV。

对比策略：
- by_dow: 按星期几分组（当前）
- by_dow_parity: 按星期几+奇偶周分组
- last_n: 最近N天均值（扫N=3,5,7,10,14）
- cycle_n: 按 day_idx % N 分组（扫N=2,3,4,5,6,7,8,10,14）
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ANOM_START = pd.Timestamp("2018-01-13")
ANOM_END = pd.Timestamp("2018-01-18 23:00")
ZONES = ["核心区", "近港区", "外围区"]


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "task_a_train.csv", encoding="utf-8-sig", parse_dates=["time_window"])
    df = df[(df["time_window"] < ANOM_START) | (df["time_window"] > ANOM_END)].copy()
    df["hour"] = df["time_window"].dt.hour
    df["day_of_week"] = df["time_window"].dt.dayofweek
    df["date"] = df["time_window"].dt.date
    return df


def cv_by_strategy(df, group_keys, value_col="vessel_count"):
    """按 group_keys 分组，计算各组跨周 CV，返回所有CV的分布统计。"""
    # 给每个时间窗口标 week_idx
    all_dates = sorted(df["date"].unique())
    date_to_week = {d: i // 7 for i, d in enumerate(all_dates)}
    df = df.copy()
    df["week_idx"] = df["date"].map(date_to_week)

    # 按 group_keys + week_idx 聚合（每周内求和/均值）
    agg = df.groupby(group_keys + ["week_idx"])[value_col].sum().reset_index()
    # 每个 group_keys 组合跨周的 CV
    cvs = []
    for keys, sub in agg.groupby(group_keys):
        vals = sub[value_col].values
        if len(vals) < 2:
            continue
        m = vals.mean()
        cv = vals.std(ddof=0) / m if m > 0 else 0
        cvs.append(cv)
    return np.array(cvs)


def print_cv_stats(name, cvs):
    if len(cvs) == 0:
        print(f"  {name}: 无足够样本")
        return
    print(f"  {name:20s}  n={len(cvs):4d}  median={np.median(cvs):.4f}  "
          f"mean={np.mean(cvs):.4f}  p25={np.percentile(cvs,25):.4f}  "
          f"p75={np.percentile(cvs,75):.4f}")


def main():
    df = load_data()
    print(f"有效数据: {df['date'].nunique()} 天, {len(df)} 行\n")

    # ===== 1. 当前策略：按星期几分组 =====
    print("=== 策略对比：按小时+分组键算跨周CV分布 ===")

    # 先加 week_idx
    all_dates_sorted = sorted(df["date"].unique())
    date_to_week = {d: i // 7 for i, d in enumerate(all_dates_sorted)}
    df["week_idx"] = df["date"].map(date_to_week)

    cvs_dow = cv_by_strategy(df, ["zone", "day_of_week", "hour"])
    print_cv_stats("by_dow (当前)", cvs_dow)

    # ===== 2. 星期几 + 奇偶周 =====
    df["week_parity"] = df["week_idx"] % 2
    cvs_dow_parity = cv_by_strategy(df, ["zone", "day_of_week", "week_parity", "hour"])
    print_cv_stats("by_dow+parity", cvs_dow_parity)

    # ===== 3. 按 day_idx % N 循环 =====
    date_to_dayidx = {d: i for i, d in enumerate(all_dates_sorted)}
    df["day_idx"] = df["date"].map(date_to_dayidx)

    print("\n--- day_idx % N 循环 ---")
    for n in [2, 3, 4, 5, 6, 7, 8, 10, 14]:
        df["cycle"] = df["day_idx"] % n
        cvs = cv_by_strategy(df, ["zone", "cycle", "hour"])
        print_cv_stats(f"cycle_{n}", cvs)

    # ===== 4. 同时看按日期聚合 =====
    print("\n--- 不按小时分组，仅按天+圈层 ---")
    cvs_daily_dow = cv_by_strategy(df, ["zone", "day_of_week"])
    print_cv_stats("daily_by_dow", cvs_daily_dow)

    for n in [2, 3, 4, 5, 6, 7, 8, 10, 14]:
        df["cycle"] = df["day_idx"] % n
        cvs = cv_by_strategy(df, ["zone", "cycle"])
        print_cv_stats(f"daily_cycle_{n}", cvs)

    # ===== 5. 最近N天均值 vs 真实值 =====
    print("\n=== 最近N天均值外推的MAE（最后7天做测试集） ===")
    test_dates = sorted(df["date"].unique())[-7:]
    train_dates = sorted(df["date"].unique())[:-7]
    test_df = df[df["date"].isin(test_dates)]
    train_df = df[df["date"].isin(train_dates)]

    for n in [3, 5, 7, 10, 14]:
        # 用训练集最后N天均值预测测试集
        last_n_dates = sorted(train_dates)[-n:]
        last_n = train_df[train_df["date"].isin(last_n_dates)]
        means = last_n.groupby(["zone", "hour"])["vessel_count"].mean()

        # 按 (zone, hour) 对齐
        test_grouped = test_df.groupby(["zone", "hour"])["vessel_count"].mean()
        common = means.index.intersection(test_grouped.index)
        mae = np.mean(np.abs(means.loc[common].values - test_grouped.loc[common].values))
        print(f"  last_{n:2d}d mean: MAE={mae:.4f}")

    # 对比：同星期同时刻均值
    dow_means = train_df.groupby(["zone", "day_of_week", "hour"])["vessel_count"].mean()
    test_df_aligned = test_df.copy()
    test_df_aligned = test_df_aligned.set_index(["zone", "day_of_week", "hour"])
    test_df_aligned["pred"] = dow_means
    test_df_aligned = test_df_aligned.dropna(subset=["pred"])
    mae_dow = np.mean(np.abs(test_df_aligned["vessel_count"] - test_df_aligned["pred"]))
    print(f"  by_dow mean: MAE={mae_dow:.4f}")


if __name__ == "__main__":
    main()