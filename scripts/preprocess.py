"""数据预处理入口脚本。

流程：
1. 加载原始训练数据
2. 清洗（过滤非拖轮、处理哨兵值）
3. 圈层分类（保留港外记录）
4. 活跃状态标注
5. 构建个体船状态表（A 题二进制 zone_state）
6. 构建代表区域表（B 题众数区域）
7. 构建赛题A/B样本
8. 补齐完整时间网格（补零）
9. 时间特征工程
10. 输出 processed/ 目录
"""
import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_training_data
from src.data.cleaner import filter_tug_vessels, clean_sentinels, remove_outlier_positions
from src.features.spatial import classify_zone
from src.features.zone import (
    label_active_rows,
    build_window_labels,
    build_vessel_state_table,
    build_vessel_repr_table,
    build_task_a_samples,
    build_task_b_samples,
)
from src.features.temporal import add_time_features, add_lag_features, add_rolling_features


# 训练期完整时间网格
TRAIN_START = "2018-01-01"
TRAIN_END = "2018-01-24"
ZONES = ["核心区", "近港区", "外围区"]
DIRECTIONS = [
    ("核心区", "近港区"), ("核心区", "外围区"),
    ("近港区", "核心区"), ("近港区", "外围区"),
    ("外围区", "核心区"), ("外围区", "近港区"),
]
ALL_HOURS = pd.date_range(f"{TRAIN_START} 00:00", f"{TRAIN_END} 23:00", freq="h")


def _fill_grid(df: pd.DataFrame, index_cols: list[str], full_index: pd.MultiIndex) -> pd.DataFrame:
    """将 df 重索引到完整时间网格上，缺失值填 0。"""
    return df.set_index(index_cols).reindex(full_index, fill_value=0).reset_index()


def expand_a_index(df: pd.DataFrame) -> pd.DataFrame:
    """补齐 A 题完整索引：24天×24h×3圈层 = 1728 行。"""
    idx = pd.MultiIndex.from_product(
        [ALL_HOURS, ZONES], names=["time_window", "zone"]
    )
    return _fill_grid(df, ["time_window", "zone"], idx)


def expand_b_index(df: pd.DataFrame) -> pd.DataFrame:
    """补齐 B 题完整索引：24天×24h×6方向 = 3456 行。"""
    tuples = [
        (tw, src, tgt)
        for tw in ALL_HOURS
        for src, tgt in DIRECTIONS
    ]
    idx = pd.MultiIndex.from_tuples(
        tuples, names=["time_window", "source_zone", "target_zone"]
    )
    return _fill_grid(df, ["time_window", "source_zone", "target_zone"], idx)


def print_ais_record_counts(df: pd.DataFrame) -> None:
    """打印每小时 AIS 条目数统计，用于识别数据断档。"""
    hourly = df.groupby(df["time_window"]).size().reset_index(name="n_records")
    hourly["date"] = hourly["time_window"].dt.date

    print("\n--- 每日 AIS 条目数（识别数据断档） ---")
    daily = (
        hourly.groupby("date")["n_records"].agg(["sum", "min", "mean"])
        .reset_index()
        .rename(columns={"sum": "total", "min": "min_hourly", "mean": "avg_hourly"})
    )
    print(daily.round(1).to_string(index=False))

    mean = daily["total"].mean()
    std = daily["total"].std()
    print(f"\n日均: {mean:.0f} ± {std:.0f}")
    low = daily[daily["total"] < mean - 2 * std]
    if len(low):
        print(f"\n异常低值日（< -2σ）：")
        for _, r in low.iterrows():
            z = (r["total"] - mean) / std
            print(f"  {r['date']}: {r['total']:.0f} 条 (z={z:.2f})")
        print(f"  正常≈{mean:.0f} 条/日，这些天仅为正常的 {low['total'].mean()/mean:.0%}")

        # 逐小时 pivot
        anom_start = low["date"].min()
        anom_end = low["date"].max()
        hourly_anom = hourly[
            (hourly["date"] >= anom_start) & (hourly["date"] <= anom_end)
        ].copy()
        hourly_anom["hour_str"] = hourly_anom["time_window"].dt.strftime("%H:00")
        pivot = hourly_anom.pivot(
            index="date", columns="hour_str", values="n_records"
        )
        print(f"\n异常期 ({anom_start} ~ {anom_end}) 逐小时记录数：")
        print(pivot.to_string())


def main():
    # 加载配置
    config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 路径
    raw_data = Path(config.get("raw_data_path", "data/raw/训练集_20180101-0124_拖轮AIS.csv"))
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载
    print("Loading data...")
    df = load_training_data(raw_data)

    # 2. 清洗
    print(f"  Raw: {len(df)} rows, {df['mmsi'].nunique()} vessels")
    df = filter_tug_vessels(df)
    print(f"  After tug filter: {len(df)} rows, {df['mmsi'].nunique()} vessels")
    df = clean_sentinels(df)
    df = remove_outlier_positions(df)
    print(f"  After cleaning: {len(df)} rows")

    # 3. 圈层分类
    print("Classifying zones...")
    df = classify_zone(
        df,
        center_lon=config["center"]["lon"],
        center_lat=config["center"]["lat"],
        radii=config["zone_radii"],
    )

    # 4. 活跃状态标注
    print("Labeling active status...")
    df["time_window"] = df["time"].dt.floor("1h")
    df = label_active_rows(df, sog_min=config["active"]["sog_min"], sog_max=config["active"]["sog_max"])
    vessel_labels = build_window_labels(df, min_records=config["active"]["min_records"])

    # 5. 构建个体船状态表（二进制 zone_state）
    print("Building vessel state table...")
    vessel_state = build_vessel_state_table(vessel_labels, df)
    vessel_state.to_csv(processed_dir / "vessel_state.csv", index=False, encoding="utf-8-sig")
    print(f"  Vessel state rows: {len(vessel_state)}")
    active_state = vessel_state[vessel_state["zone_state"] > 0]
    print(f"  Active (zone_state>0): {len(active_state)} rows")
    for zone, bit in [("核心区", 4), ("近港区", 2), ("外围区", 1)]:
        n = (active_state["zone_state"] & bit).gt(0).sum()
        print(f"    {zone}: {n} vessel-hours")
    multi = active_state["zone_state"].apply(lambda s: bin(s).count("1") > 1)
    print(f"    多圈层同时活跃: {multi.sum()} vessel-hours")

    # 5b. AIS 条目数统计（数据断档检测）
    print_ais_record_counts(df)

    # 保存每小时 AIS 条目数，供下游分析/画图使用
    hourly_counts = df.groupby(df["time_window"]).size().reset_index(name="n_records")
    hourly_counts.to_csv(processed_dir / "ais_record_counts.csv", index=False, encoding="utf-8-sig")

    # 6. 构建 B 题代表区域表（按全部记录众数 + 时间并列规则）
    print("Building vessel representative zone table...")
    repr_table = build_vessel_repr_table(df)
    repr_table.to_csv(processed_dir / "vessel_repr.csv", index=False, encoding="utf-8-sig")
    print(f"  Representative zone rows: {len(repr_table)}")

    # 7. 构建聚合赛题样本
    print("Building task samples...")
    task_a = build_task_a_samples(vessel_state)
    task_b = build_task_b_samples(repr_table)

    # 8. 补齐完整时间网格（补零），保证时序特征正确
    print("Expanding to full time grid...")
    task_a = expand_a_index(task_a)
    task_b = expand_b_index(task_b)

    # 9. 时间特征
    print("Adding temporal features...")
    task_a = add_time_features(task_a)
    task_a = add_lag_features(task_a, group_cols=["zone"])
    task_a = add_rolling_features(task_a, group_cols=["zone"])

    task_b = add_time_features(task_b)
    task_b = add_lag_features(task_b, group_cols=["source_zone", "target_zone"])
    task_b = add_rolling_features(task_b, group_cols=["source_zone", "target_zone"])

    # 10. 保存
    print("Saving processed data...")
    task_a.to_csv(processed_dir / "task_a_train.csv", index=False, encoding="utf-8-sig")
    task_b.to_csv(processed_dir / "task_b_train.csv", index=False, encoding="utf-8-sig")
    print(f"  Task A samples: {len(task_a)}")
    print(f"  Task B samples: {len(task_b)}")
    print("Done.")


if __name__ == "__main__":
    main()
