"""训练与预测 v2：两步解耦策略。

基于探索结论：
- 日总量跨周 CV 仅 0.10-0.14（远低于小时级 0.35+）
- 最近 N 天均值外推远好于同星期同时刻均值

策略：
  Step 1: 预测每日每圈层/方向的总量（用最近 N 天均值）
  Step 2: 按历史小时比例分配到每小时
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.submission import generate_submissions

PROCESSED_DIR = Path("data/processed")
TEMPLATE_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/submission")

GAP_START = pd.Timestamp("2018-01-13")
GAP_END = pd.Timestamp("2018-01-18 23:00")
PREDICT_START = pd.Timestamp("2018-01-25")
PREDICT_END = pd.Timestamp("2018-01-31 23:00")
N_DAYS = 10  # 用于预测日总量的最近天数

ZONES = ["核心区", "近港区", "外围区"]
DIRECTIONS = [
    ("核心区", "近港区"), ("核心区", "外围区"),
    ("近港区", "核心区"), ("近港区", "外围区"),
    ("外围区", "核心区"), ("外围区", "近港区"),
]


def load_and_filter(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / name, encoding="utf-8-sig", parse_dates=["time_window"])
    mask = (df["time_window"] < GAP_START) | (df["time_window"] > GAP_END)
    df = df[mask].copy()
    df["date"] = df["time_window"].dt.date
    df["hour"] = df["time_window"].dt.hour
    return df


def predict_daily_total(daily_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """用最近 N 天均值预测未来 7 天每日总量。

    Parameters
    ----------
    daily_df : 含 date, group_cols, value 列
    group_cols : 分组键（如 ["zone"] 或 ["source_zone", "target_zone"]）

    Returns
    -------
    pd.DataFrame，含 date, group_cols, pred_daily 列
    """
    all_dates = sorted(daily_df["date"].unique())
    last_n_dates = all_dates[-N_DAYS:]
    recent = daily_df[daily_df["date"].isin(last_n_dates)]

    # 每组最近 N 天均值
    daily_mean = recent.groupby(group_cols)["value"].mean().reset_index(name="pred_daily")

    # 生成未来 7 天
    pred_dates = pd.date_range(PREDICT_START, PREDICT_END, freq="D").date
    rows = []
    for d in pred_dates:
        for _, r in daily_mean.iterrows():
            row = {"date": d, "pred_daily": r["pred_daily"]}
            for i, col in enumerate(group_cols):
                row[col] = r[col]
            rows.append(row)
    return pd.DataFrame(rows)


def compute_hourly_proportion(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """计算每组的历史小时分布比例。

    Returns
    -------
    pd.DataFrame，含 group_cols, hour, proportion 列（每组 proportion 和为 1）
    """
    hourly = df.groupby(group_cols + ["hour"])["value"].sum().reset_index(name="total")
    daily = hourly.groupby(group_cols)["total"].transform("sum")
    hourly["proportion"] = hourly["total"] / daily.replace(0, np.nan)
    hourly["proportion"] = hourly["proportion"].fillna(0)
    return hourly[group_cols + ["hour", "proportion"]]


def distribute_to_hours(daily_pred: pd.DataFrame, hourly_prop: pd.DataFrame,
                        group_cols: list[str]) -> pd.DataFrame:
    """将日总量预测按历史比例分配到每小时。"""
    pred_dates = sorted(daily_pred["date"].unique())
    pred_hours = pd.date_range(
        pd.Timestamp(pred_dates[0]), pd.Timestamp(pred_dates[-1]) + pd.Timedelta(hours=23), freq="h"
    )
    rows = []
    for tw in pred_hours:
        d = tw.date()
        h = tw.hour
        day_pred = daily_pred[daily_pred["date"] == d]
        for _, r in day_pred.iterrows():
            key = {}
            for col in group_cols:
                key[col] = r[col]
            # 找对应比例
            prop_mask = True
            for col in group_cols:
                prop_mask = prop_mask & (hourly_prop[col] == r[col])
            prop_mask = prop_mask & (hourly_prop["hour"] == h)
            prop = hourly_prop[prop_mask]["proportion"]
            prop_val = prop.iloc[0] if len(prop) > 0 else 0
            pred_val = max(0, round(r["pred_daily"] * prop_val))
            row = {"time_window": tw, "predicted": pred_val}
            for col in group_cols:
                row[col] = r[col]
            rows.append(row)
    return pd.DataFrame(rows)


def task_a_predict(df: pd.DataFrame) -> pd.DataFrame:
    """A题：预测 3圈层 × 168小时 的活跃拖轮数。"""
    # 日总量
    daily = df.groupby(["date", "zone"])["vessel_count"].sum().reset_index(name="value")
    daily_pred = predict_daily_total(daily, ["zone"])

    # 小时比例
    hourly_prop = compute_hourly_proportion(df.rename(columns={"vessel_count": "value"}), ["zone"])

    # 分配
    return distribute_to_hours(daily_pred, hourly_prop, ["zone"])


def task_b_predict(df: pd.DataFrame) -> pd.DataFrame:
    """B题：预测 6方向 × 168小时 的迁移量。"""
    # 日总量
    daily = df.groupby(["date", "source_zone", "target_zone"])["vessel_count"].sum().reset_index(name="value")
    daily_pred = predict_daily_total(daily, ["source_zone", "target_zone"])

    # 小时比例
    hourly_prop = compute_hourly_proportion(
        df.rename(columns={"vessel_count": "value"}),
        ["source_zone", "target_zone"],
    )

    return distribute_to_hours(daily_pred, hourly_prop, ["source_zone", "target_zone"])


def main():
    print("=== 加载数据 ===")
    a_train = load_and_filter("task_a_train.csv")
    b_train = load_and_filter("task_b_train.csv")
    print(f"A题训练: {len(a_train)} 行, {a_train['date'].nunique()} 天")
    print(f"B题训练: {len(b_train)} 行, {b_train['date'].nunique()} 天")

    # 打印日总量统计
    print("\n--- A题 日总量（最近10天） ---")
    a_daily = a_train.groupby(["date", "zone"])["vessel_count"].sum().reset_index()
    all_dates = sorted(a_daily["date"].unique())
    for zone in ZONES:
        recent = a_daily[(a_daily["date"].isin(all_dates[-N_DAYS:])) & (a_daily["zone"] == zone)]
        print(f"  {zone}: mean={recent['vessel_count'].mean():.1f}, std={recent['vessel_count'].std():.1f}")

    print("\n--- B题 日总量（最近10天） ---")
    b_daily = b_train.groupby(["date", "source_zone", "target_zone"])["vessel_count"].sum().reset_index()
    for src, tgt in DIRECTIONS:
        recent = b_daily[
            (b_daily["date"].isin(all_dates[-N_DAYS:]))
            & (b_daily["source_zone"] == src)
            & (b_daily["target_zone"] == tgt)
        ]
        print(f"  {src}->{tgt}: mean={recent['vessel_count'].mean():.1f}")

    print("\n=== 预测 ===")
    a_pred = task_a_predict(a_train)
    b_pred = task_b_predict(b_train)

    print(f"  A题预测: {len(a_pred)} 行")
    print(f"  B题预测: {len(b_pred)} 行")

    print("\n=== 生成提交文件 ===")
    paths = generate_submissions(a_pred, b_pred, TEMPLATE_DIR, OUTPUT_DIR)
    print(f"  A题: {paths[0]}")
    print(f"  B题: {paths[1]}")

    # 打印摘要
    print("\n--- A题 预测摘要 ---")
    for zone in ZONES:
        total = a_pred[a_pred["zone"] == zone]["predicted"].sum()
        print(f"  {zone}: 7天总预测={total}, 日均={total/7:.1f}")

    print("\n--- B题 预测摘要 ---")
    for src, tgt in DIRECTIONS:
        sub = b_pred[(b_pred["source_zone"] == src) & (b_pred["target_zone"] == tgt)]
        total = sub["predicted"].sum()
        print(f"  {src}->{tgt}: 7天总预测={total}, 日均={total/7:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()