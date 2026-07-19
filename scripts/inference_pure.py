"""
CTS2026 — 纯推理（零依赖 TSLib，直接调源库）
Usage:
    python scripts/inference_pure.py --task a --model moirai --device cuda
    python scripts/inference_pure.py --task a --model chronos --device cuda
    python scripts/inference_pure.py --task a --model aurora --device cpu

Dependencies:
    moirai        : pip install uni2ts
    chronos       : pip install chronos
    aurora        : pip install aurora-ts  (HF 自动下载) 或指定 --aurora_ckpt 本地权重

Models are cached to data/models/ (configurable via --model_dir).
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJ_ROOT, "data", "models")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "data", "submission")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_path, seq_len):
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df.rename(columns={"time_window": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = [c for c in df.columns if c != "date"]
    data = df[feature_cols].values.astype(np.float32)
    context = data[-seq_len:]
    last_date = df["date"].iloc[-1]
    return context, feature_cols, last_date


def make_future_dates(last_date, pred_len, freq="h"):
    return pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=pred_len, freq=freq)


# ---------------------------------------------------------------------------
# Moirai (source: uni2ts)
# ---------------------------------------------------------------------------
def predict_moirai(context, args):
    try:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    except ImportError:
        sys.exit("Moirai requires: pip install uni2ts")
    import torch

    device = torch.device(args.device)
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(args.moirai_id, cache_dir=args.model_dir),
        prediction_length=args.pred_len,
        context_length=args.seq_len,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    model.eval()

    x = torch.FloatTensor(context).unsqueeze(0).to(device)  # [1, L, C]
    C = x.shape[-1]
    outputs = []
    with torch.no_grad():
        for i in range(C):
            out = model.predict(x[:, :, i].squeeze(0).cpu().numpy())
            out = np.mean(out, axis=1)  # avg over samples
            outputs.append(torch.from_numpy(out).float())
    return torch.stack(outputs, dim=-1).numpy()  # [pred_len, C]


# ---------------------------------------------------------------------------
# Chronos (source: chronos)
# ---------------------------------------------------------------------------
def predict_chronos(context, args):
    try:
        from chronos import BaseChronosPipeline
    except ImportError:
        sys.exit("Chronos requires: pip install chronos")
    import torch

    model = BaseChronosPipeline.from_pretrained(
        args.chronos_id,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        cache_dir=args.model_dir,
    )

    x = torch.FloatTensor(context).unsqueeze(0)  # [1, L, C]
    C = x.shape[-1]
    outputs = []
    for i in range(C):
        out = model.predict(x[:, :, i], prediction_length=args.pred_len)  # (1, n_samples, pred_len)
        out = out.mean(dim=1).squeeze(0)  # (pred_len,)
        outputs.append(out)
    return torch.stack(outputs, dim=-1).numpy()  # (pred_len, C)


# ---------------------------------------------------------------------------
# Aurora (source: aurora-ts)
# ---------------------------------------------------------------------------
def _build_aurora_from_local(ckpt_path):
    from aurora.modeling_aurora import AuroraForPrediction
    from aurora.configuration_aurora import AuroraConfig
    from aurora.utils.path_utils import get_package_file_path

    config = AuroraConfig.from_json_file(get_package_file_path("config.json"))
    model = AuroraForPrediction(config)
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safetensors_load_file
        weights = safetensors_load_file(ckpt_path, device="cpu")
    else:
        weights = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


def predict_aurora(context, args):
    try:
        from aurora import load_model as aurora_load
    except ImportError:
        sys.exit("Aurora requires: pip install aurora-ts")
    import torch

    ckpt = args.aurora_ckpt.strip() or os.environ.get("AURORA_CKPT", "")
    if not ckpt:
        local_ckpt = os.path.join(args.model_dir, "aurora", "model.safetensors")
        if os.path.isfile(local_ckpt):
            ckpt = local_ckpt
    if ckpt:
        ckpt = os.path.expanduser(ckpt)
        if os.path.isdir(ckpt):
            ckpt = os.path.join(ckpt, "model.safetensors")
        print(f"  [Aurora] loading local: {ckpt}")
        model = _build_aurora_from_local(ckpt)
    else:
        model = aurora_load()
    model.eval()

    device = torch.device(args.device)
    model = model.to(device)
    x = torch.FloatTensor(context).unsqueeze(0).to(device)  # [1, L, C]
    B, L, C = x.shape

    # channel-independent: (B, L, C) → (B*C, L)
    flat = x.permute(0, 2, 1).reshape(B * C, L)

    with torch.no_grad():
        out = model.generate(
            inputs=flat,
            max_output_length=args.pred_len,
            num_samples=args.num_samples,
            inference_token_len=args.inference_token_len,
        )  # (B*C, num_samples, pred_len)
    out = out.mean(dim=1)  # (B*C, pred_len) — avg over samples
    out = out.reshape(B, C, args.pred_len).permute(0, 2, 1).contiguous()  # (B, pred_len, C)
    return out.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PREDICTORS = {
    "moirai":            predict_moirai,
    "chronos":           predict_chronos,
    "aurora":            predict_aurora,
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def wide_to_submission(df, task):
    """宽表 → 赛题提交格式（长表）"""
    df["time_window"] = pd.to_datetime(df["time_window"])

    if task == "a":
        long_df = df.melt(id_vars=["time_window"], var_name="zone", value_name="vessel_count")
        long_df = long_df.sort_values(["time_window", "zone"]).reset_index(drop=True)
        return long_df[["time_window", "zone", "vessel_count"]]

    # Task B: melt + 拆 source/target
    long_df = df.melt(id_vars=["time_window"], var_name="direction", value_name="vessel_count")
    long_df[["source_zone", "target_zone"]] = long_df["direction"].str.split("→", expand=True)
    long_df = long_df.drop(columns=["direction"])
    long_df = long_df.sort_values(["time_window", "source_zone", "target_zone"]).reset_index(drop=True)
    return long_df[["time_window", "source_zone", "target_zone", "vessel_count"]]


def main():
    parser = argparse.ArgumentParser(description="CTS2026 — Pure Inference (no TSLib)")
    parser.add_argument("--task", type=str, default="a", choices=["a", "b"])
    parser.add_argument("--model", type=str, default="chronos")
    parser.add_argument("--seq_len", type=int, default=576)
    parser.add_argument("--pred_len", type=int, default=168)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="",
                        help="custom wide table path (default: data/processed/task_X_train_wide.csv)")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR,
                        help="local cache for HF models (set HF_HUB_CACHE)")
    # Moirai
    parser.add_argument("--moirai_id", type=str, default="Salesforce/moirai-2.0-R-small")
    # Chronos
    parser.add_argument("--chronos_id", type=str, default="amazon/chronos-bolt-base")
    # Aurora
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--inference_token_len", type=int, default=48)
    parser.add_argument("--aurora_ckpt", type=str, default="")
    args = parser.parse_args()

    if args.model_dir:
        os.environ["HF_HUB_CACHE"] = os.path.abspath(args.model_dir)
        os.makedirs(args.model_dir, exist_ok=True)

    data_path = args.data or os.path.join(DATA_DIR, f"task_{args.task}_train_wide.csv")
    print(f"[ CTS2026 Task {args.task.upper()} ]  model={args.model}  "
          f"seq_len={args.seq_len}  pred_len={args.pred_len}  device={args.device}")
    print(f"  data: {data_path}")

    context, feature_cols, last_date = load_data(data_path, args.seq_len)
    future_dates = make_future_dates(last_date, args.pred_len)
    print(f"  context: {len(context)} steps, ends {last_date}")
    print(f"  forecast: {future_dates[0]} ~ {future_dates[-1]}")

    predictor = PREDICTORS.get(args.model)
    if predictor is None:
        sys.exit(f"Unknown model: {args.model}. Available: {list(PREDICTORS.keys())}")

    pred = predictor(context, args)
    pred = np.round(np.clip(pred, 0, None)).astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_wide = pd.DataFrame(pred, columns=feature_cols)
    out_wide.insert(0, "time_window", future_dates)
    wide_path = os.path.join(OUTPUT_DIR, f"task_{args.task}_pred_{args.model}_wide.csv")
    out_wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    out_sub = wide_to_submission(out_wide, args.task)
    sub_path = os.path.join(OUTPUT_DIR, f"task_{args.task}_pred_{args.model}.csv")
    out_sub.to_csv(sub_path, index=False, encoding="utf-8-sig")
    print(f"\n  saved → {sub_path}  ({len(out_sub)} rows)")
    print(f"  saved → {wide_path}  ({len(out_wide)} rows)\n")
    print(out_sub.to_string(index=False))


if __name__ == "__main__":
    main()
