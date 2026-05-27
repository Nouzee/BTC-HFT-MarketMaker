import os
import sys
import polars as pl
from pathlib import Path

# Add code directory to path for imports
code_dir = Path(__file__).parent.parent
sys.path.append(str(code_dir))
from engine.maker_hft_backtest import run_standard_oos


def main() -> None:
    # 1) Configure data directory
    # default: data directory under project root
    default_data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir = Path(os.getenv("HFT_DATA_DIR", default_data_dir)).expanduser()
    feature_file = data_dir / "events_features.parquet"

    # 2) Validate feature parquet
    if not feature_file.exists():
        raise FileNotFoundError(f"特征文件不存在: {feature_file}")

    # 3) 70% / 30% OOS time split (true out-of-sample evaluation)
    events = pl.read_parquet(feature_file)
    ts_min = events["ts_ms"].min()
    ts_max = events["ts_ms"].max()
    split_point = ts_min + int(0.7 * (ts_max - ts_min))

    oos_results = run_standard_oos(
        cfg=None,
        feature_parquet=feature_file,
        train_start=int(ts_min),
        train_end=int(split_point),
        test_start=int(split_point),
        test_end=int(ts_max),
    )

    # 4) Print metrics
    print("=== OOS Train Metrics ===")
    for k, v in oos_results["train"].items():
        print(f"{k}: {v}")

    print("\n=== OOS Test Metrics ===")
    for k, v in oos_results["test"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
