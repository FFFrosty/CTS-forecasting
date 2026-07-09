"""数据预处理入口脚本。

流程：
1. 加载原始训练数据
2. 清洗（过滤非拖轮、处理哨兵值）
3. 圈层分类
4. 活跃状态标注
5. 构建赛题A/B样本
6. 时间特征工程
7. 输出 processed/ 目录
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
    build_task_a_samples,
    build_task_b_samples,
)
from src.features.temporal import add_time_features, add_lag_features, add_rolling_features


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

    # 5. 构建赛题样本
    print("Building task samples...")
    task_a = build_task_a_samples(vessel_labels, df)
    task_b = build_task_b_samples(vessel_labels, df)

    # 6. 时间特征
    print("Adding temporal features...")
    task_a = add_time_features(task_a)
    task_a = add_lag_features(task_a, group_cols=["zone"])
    task_a = add_rolling_features(task_a, group_cols=["zone"])

    task_b = add_time_features(task_b)
    task_b = add_lag_features(task_b, group_cols=["source_zone", "target_zone"])
    task_b = add_rolling_features(task_b, group_cols=["source_zone", "target_zone"])

    # 7. 保存
    print("Saving processed data...")
    task_a.to_csv(processed_dir / "task_a_train.csv", index=False, encoding="utf-8-sig")
    task_b.to_csv(processed_dir / "task_b_train.csv", index=False, encoding="utf-8-sig")
    print(f"  Task A samples: {len(task_a)}")
    print(f"  Task B samples: {len(task_b)}")
    print("Done.")


if __name__ == "__main__":
    main()
