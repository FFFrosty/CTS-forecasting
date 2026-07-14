"""可视化：每天一张角色分布热力图，分A/B题口径。

A题 → role_daily_a/：底色=zone_state (-1~7), 数字=zone_state值(>0才标)
B题 → role_daily_b/：底色=repr_zone (-1~4), 纯色块无数字

-1=无数据(白), 0=不活跃/港外(灰), 正整数=活跃/圈层
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_A = PROCESSED_DIR / "viz" / "role_daily_a"
OUT_B = PROCESSED_DIR / "viz" / "role_daily_b"
OUT_A.mkdir(parents=True, exist_ok=True)
OUT_B.mkdir(parents=True, exist_ok=True)

DOW_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
GAP_DATE_STRS = {str(d.date()) for d in pd.date_range("2018-01-13", "2018-01-18")}

# ===== A 题：zone_state (-1=无数据, 0=无活跃, 1~7=活跃组合) =====
STATE_COLORS = [
    "#ffffff",  # -1 无数据 —— 白
    "#e0e0e0",  #  0 无活跃 —— 浅灰
    "#fff176",  #  1 仅外围 —— 黄
    "#ff9800",  #  2 仅近港 —— 橙
    "#e6a817",  #  3 外围+近港 —— 棕黄
    "#e53935",  #  4 仅核心 —— 红
    "#f06292",  #  5 核心+外围 —— 粉红
    "#d84315",  #  6 核心+近港 —— 深橙
    "#880e4f",  #  7 全部三区 —— 深红
]
# 文字颜色：深底色用白字，浅底色用黑字
STATE_TEXT_COLORS = {
    -1: "#aaaaaa", 0: "#999999",
    1: "#000000", 2: "#000000", 3: "#000000",
    4: "#ffffff", 5: "#ffffff", 6: "#ffffff", 7: "#ffffff",
}
cmap_state = ListedColormap(STATE_COLORS)
norm_state = BoundaryNorm(boundaries=np.arange(-1.5, 8.5, 1), ncolors=9)

# ===== B 题：repr_zone (-1=无数据, 0=港外, 1~4=三圈层) =====
ZONE_CODE = {"无数据": -1, "港外": 0, "外围区": 1, "近港区": 2, "核心区": 3}
ZONE_COLORS = [
    "#ffffff",  # -1 无数据 —— 白
    "#cccccc",  #  0 港外 —— 灰
    "#fff176",  #  1 外围区 —— 黄
    "#ff9800",  #  2 近港区 —— 橙
    "#e53935",  #  3 核心区 —— 红
]
ZONE_WEIGHT = {"核心区": 3, "近港区": 2, "外围区": 1, "港外": 0, "无数据": -1}
cmap_zone = ListedColormap(ZONE_COLORS)
norm_zone = BoundaryNorm(boundaries=np.arange(-1.5, 4.5, 1), ncolors=5)


def fill_24h(df, extra_defaults):
    """补全每船每天 24 小时网格，缺失小时填默认值。"""
    existing = set(map(tuple, df[["mmsi", "date", "hour"]].drop_duplicates().values))
    missing = []
    for m, d in df[["mmsi", "date"]].drop_duplicates().itertuples(index=False):
        for h in range(24):
            if (m, d, h) not in existing:
                missing.append({"mmsi": m, "date": d, "hour": h, **extra_defaults})
    if missing:
        df = pd.concat([df, pd.DataFrame(missing)], ignore_index=True)
    return df


def draw_day(date, day_group, state, value_col, out_dir, cmap, norm, title_suffix,
             text_colors=None):
    """绘制一天热力图。"""
    n_vessels = len(day_group)
    matrix = np.zeros((n_vessels, 24), dtype=int)
    mmsi_labels = []

    for i, m in enumerate(day_group["mmsi"]):
        one = state[(state["mmsi"] == m) & (state["date"] == date)].sort_values("hour")
        matrix[i] = one[value_col].fillna(-1).astype(int).values
        mmsi_labels.append(str(m))

    fig, ax = plt.subplots(figsize=(12, max(8, n_vessels * 0.3)))
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # 格子内标数字
    if text_colors is not None:
        for i in range(n_vessels):
            for j in range(24):
                val = matrix[i, j]
                if val > 0:  # 只标活跃状态
                    c = text_colors.get(val, "#000000")
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=7, color=c, fontweight="bold")

    ylabels = [f"{mmsi_labels[i]} ({day_group.iloc[i]['score']:.1f})" for i in range(n_vessels)]
    ax.set_yticks(range(n_vessels))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_xlabel("小时")

    is_gap = str(date) in GAP_DATE_STRS
    bg_note = ""
    if is_gap:
        ax.axhspan(-0.5, n_vessels - 0.5, color="red", alpha=0.06, zorder=0)
        bg_note = " [断层期]"
    elif date.weekday() in [5, 6]:
        ax.axhspan(-0.5, n_vessels - 0.5, color="orange", alpha=0.04, zorder=0)
        bg_note = " [周末]"

    s_min, s_max = day_group["score"].min(), day_group["score"].max()
    ax.set_title(
        f"{date.strftime('%Y-%m-%d')} {DOW_NAMES[date.weekday()]}{bg_note}  "
        f"({n_vessels}艘, score {s_min:.1f}~{s_max:.1f}){title_suffix}"
    )

    plt.tight_layout()
    plt.savefig(out_dir / f"{date}.png", dpi=120, bbox_inches="tight")
    plt.close()


def main():
    # ===== A 题：zone_state =====
    print("=== A 题 ===")
    sa = pd.read_csv(PROCESSED_DIR / "vessel_state.csv", parse_dates=["time_window"])
    sa["date"] = sa["time_window"].dt.date
    sa["hour"] = sa["time_window"].dt.hour
    print(f"  {len(sa)} 行, {sa['mmsi'].nunique()} 艘船")

    sa = fill_24h(sa, {"zone_state": -1})
    # 排序：只对真实数据（zone_state>=0）算均值
    scores_a = (
        sa[sa["zone_state"] >= 0].groupby(["mmsi", "date"])["zone_state"]
        .mean().reset_index(name="score")
    )
    # 补回只在 -1 状态出现的船
    all_pairs = sa[["mmsi", "date"]].drop_duplicates()
    scores_a = all_pairs.merge(scores_a, on=["mmsi", "date"], how="left")
    scores_a["score"] = scores_a["score"].fillna(-1)

    all_dates = sorted(sa["date"].unique())
    for date in all_dates:
        day = scores_a[scores_a["date"] == date].sort_values("score", ascending=False)
        draw_day(date, day, sa, "zone_state", OUT_A, cmap_state, norm_state,
                 text_colors=STATE_TEXT_COLORS,
                 title_suffix="\n[ A题 ] 白=无数据 灰=无活跃 黄=外围 橙=近港 红=核心 粉=核+外 深橙=核+近 深红=全部")

    print(f"A题完成: {len(all_dates)} 张图 → {OUT_A}")

    # ===== B 题：repr_zone =====
    print("=== B 题 ===")
    sb = pd.read_csv(PROCESSED_DIR / "vessel_repr.csv", parse_dates=["time_window"])
    sb["date"] = sb["time_window"].dt.date
    sb["hour"] = sb["time_window"].dt.hour
    sb["repr_code"] = sb["repr_zone"].map(ZONE_CODE).fillna(-1).astype(int)
    print(f"  {len(sb)} 行, {sb['mmsi'].nunique()} 艘船")

    sb = fill_24h(sb, {"repr_zone": "无数据", "repr_code": -1})
    sb["zone_weight"] = sb["repr_zone"].map(ZONE_WEIGHT).fillna(-1)
    scores_b = sb.groupby(["mmsi", "date"])["zone_weight"].mean().reset_index(name="score")

    all_dates_b = sorted(sb["date"].unique())
    for date in all_dates_b:
        day = scores_b[scores_b["date"] == date].sort_values("score", ascending=False)
        draw_day(date, day, sb, "repr_code", OUT_B, cmap_zone, norm_zone,
                 text_colors=None,
                 title_suffix="\n[ B题 ] 白=无数据 灰=港外 黄=外围 橙=近港 红=核心")

    print(f"B题完成: {len(all_dates_b)} 张图 → {OUT_B}")


if __name__ == "__main__":
    main()