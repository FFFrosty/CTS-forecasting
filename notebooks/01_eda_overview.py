# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
# %% [markdown]
# # CTS 2026 — 拖轮AIS数据探索性分析
#
# 目标：理解数据分布、发现质量问题、为特征工程提供方向。

# %% Imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent if "__file__" not in dir() else Path(__file__).parent.parent))

from src.data.loader import load_training_data
from src.data.cleaner import filter_tug_vessels, clean_sentinels

# %% 加载数据
DATA_PATH = "data/raw/训练集_20180101-0124_拖轮AIS.csv"
df = load_training_data(DATA_PATH)
print(f"总行数: {len(df):,}")
print(f"唯一MMSI: {df['mmsi'].nunique()}")
print(f"时间范围: {df['time'].min()} ~ {df['time'].max()}")
print(f"\n列: {list(df.columns)}")
print(f"\n数据类型:\n{df.dtypes}")

# %% 船型分布
print("=== 船型分布 ===")
type_counts = df["ship_type"].value_counts()
for t, c in type_counts.items():
    print(f"  {t}: {c:,}")

# %% 清洗后数据
df_clean = filter_tug_vessels(df)
df_clean = clean_sentinels(df_clean)
print(f"清洗后行数: {len(df_clean):,}")
print(f"清洗后MMSI: {df_clean['mmsi'].nunique()}")

# %% 数值列统计
num_cols = ["x", "y", "cog", "true_heading", "sog", "rot"]
print(df_clean[num_cols].describe())

# %% 缺失率
print("\n=== 缺失率 ===")
for col in num_cols:
    na_rate = df_clean[col].isna().mean()
    print(f"  {col}: {na_rate:.2%}")

# %% SOG 分布
df_clean["sog"].hist(bins=100, figsize=(10, 4))
plt.axvline(x=2, color="r", linestyle="--", label="active_min (2kt)")
plt.axvline(x=10, color="r", linestyle="--", label="active_max (10kt)")
plt.title("SOG Distribution")
plt.xlabel("Speed Over Ground (knots)")
plt.legend()
plt.show()

# %% 每日船舶数量趋势
df_clean["date"] = df_clean["time"].dt.date
daily_vessels = df_clean.groupby("date")["mmsi"].nunique()
daily_vessels.plot(figsize=(12, 4), marker="o")
plt.title("Daily Unique Vessel Count (Training Set)")
plt.xlabel("Date")
plt.ylabel("Vessel Count")
plt.grid(True, alpha=0.3)
plt.show()

# %% 每小时记录数分布
df_clean["hour"] = df_clean["time"].dt.hour
hourly_records = df_clean.groupby("hour").size()
hourly_records.plot(kind="bar", figsize=(12, 4))
plt.title("Records per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Record Count")
plt.show()

# %% 空间分布
plt.figure(figsize=(10, 8))
sample = df_clean.sample(min(50000, len(df_clean)))
plt.scatter(sample["x"], sample["y"], s=1, alpha=0.3)
plt.scatter([117.79], [38.97], c="red", s=100, marker="*", label="Port Center")
# 圈层示意
from matplotlib.patches import Circle
ax = plt.gca()
for r, label, color in [(3, "Core", "red"), (10, "Near Port", "orange"), (30, "Outer", "green")]:
    # 粗略：1° ≈ 111km
    circle = Circle((117.79, 38.97), r/111, fill=False, color=color, linestyle="--", label=label)
    ax.add_patch(circle)
plt.legend()
plt.title("Spatial Distribution of AIS Records")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("equal")
plt.show()

print("\nEDA 完成。")
