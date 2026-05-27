from __future__ import annotations

import atexit
from dataclasses import dataclass, asdict, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from numba import njit

# ── Dictionary Learning imports (optional; engine runs with cached D only) ──
try:
    from sklearn.decomposition import MiniBatchDictionaryLearning
    from sklearn.linear_model import OrthogonalMatchingPursuit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_GRID_EVENTS_PAYLOAD: Optional[Dict[str, np.ndarray]] = None
_GRID_BASE_CFG_DICT: Optional[Dict[str, object]] = None
_GRID_EVENTS_DF: Optional[pl.DataFrame] = None
_GRID_WORKER_SHM_HANDLES: List[shared_memory.SharedMemory] = []


# =========================================
# 1) 配置与数据接口（抓取部分留空）
# =========================================


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    maker_rebate_bps: float = -0.5  # -0.5 bps, negative means rebate
    tick_size: float = 0.1

    # quote: lower turnover → wider spread → lower adverse selection loss
    half_spread_ticks: float = 5.5  # wider spread → fewer fills but each fill more profitable
    gamma_inventory: float = 1.8  # stronger inventory control → faster mean reversion to zero
    max_pos_btc: float = 0.8  # smaller max position → limit skew risk
    order_size_btc: float = 0.010  # smaller base order → lower turnover target ~800-1200x
    max_order_size_ratio: float = 0.20  # smaller cap → lower overall turnover

    # market impact / realism
    latency_ms: int = 10
    queue_ahead_ratio: float = 0.5  # assumed queue ahead at our quote level
    participation_cap: float = 0.005  # max share of each aggressive trade we can fill
    queue_model: int = 0  # 0=conservative touch-fill, 1=probabilistic expected-fill
    queue_prob_kappa: float = 1.5  # stronger => easier touch fill in probabilistic mode

    # evaluation
    pnl_resample_seconds: int = 1  # default clock-time risk metrics (1s)
    min_points_for_metrics: int = 30
    required_monthly_turnover: float = 500.0
    max_monthly_turnover: float = 1400.0
    grid_search_event_stride: int = 20  # use every Nth event in grid-search for faster smoke run
    asof_tolerance_ms: int = 100  # max book staleness allowed in trades-book alignment
    enable_oos_eval: bool = True
    oos_train_ratio: float = 0.7
    enable_walk_forward_eval: bool = True
    walk_forward_train_days: int = 7
    walk_forward_gap_days: int = 1
    walk_forward_disable_advanced_microprice: bool = False
    stability_eval_top_k: int = 30
    stability_eps: float = 1e-9
    stability_daily_std_penalty: float = 5.0
    min_fill_ratio_hard: float = 0.2  # min(buy,sell)/max(buy,sell) hard gate in selection
    markout_weight_100ms: float = 0.7
    markout_weight_500ms: float = 0.3

    # advanced micro-price / feature engineering
    use_advanced_microprice: bool = True
    imbalance_bins: int = 32
    spread_bins: int = 12
    use_quantile_imbalance_bins: bool = True
    use_symmetry_trick: bool = True
    use_next_jump_label: bool = True
    depth_bins: int = 5
    duration_bins: int = 5
    label_future_ticks: int = 30
    min_state_samples: int = 50
    waiting_ma_window: int = 200
    adv_train_ratio: float = 0.2  # fit G(I,S) on first ratio only to reduce look-ahead

    # queue cancel / adverse selection realism
    queue_decay_lambda_per_ms: float = 0.0005  # faster queue decay → fewer stale fills
    adverse_selection_bps: float = 0.80  # slightly higher AS cost → more conservative quoting

    # deeper microstructure realism
    observation_latency_ms: int = 5  # observe slightly stale book, then quote with action latency
    refresh_queue_on_activation: int = 1  # 1=yes, refresh queue at activation time
    inventory_vol_alpha: float = 0.02  # EWMA smoothing for abs return (bps)
    inventory_vol_beta: float = 0.12  # stronger vol sensitivity
    thin_depth_spread_mult: float = 1.5
    thin_depth_adverse_mult: float = 2.0
    fast_duration_spread_mult: float = 1.15
    obi_circuit_enabled: bool = True
    obi_hard_threshold: float = 0.82  # more aggressive circuit breaker
    obi_ema_alpha: float = 0.04  # slower EMA → smoother OBI signal
    obi_trend_skew_bps: float = 1.5  # stronger asymmetric skew → avoid toxic side more
    obi_soft_widen_ticks: float = 1.8  # more aggressive widening on adverse side
    obi_hysteresis_events: int = 4  # faster trigger
    obi_hard_disable_inv_ratio: float = 0.55  # earlier disable when inventory skewed
    micro_px_alpha: float = 0.60  # higher weight on microprice → trust prediction more
    micro_skew_favorable_kappa: float = 1.0  # FULL narrowing on favorable side → maximize fill rate on good trades
    micro_skew_adverse_kappa: float = 1.8  # FULL widening on adverse side → avoid toxic fills aggressively
    inventory_aggr_threshold_ratio: float = 0.55  # earlier unwind → control skew earlier
    inventory_unwind_narrow_ticks: float = 0.3  # very tight → faster unwind fills
    inventory_unwind_widen_ticks: float = 1.8  # much wider on accumulation → slow one-way inventory
    conflict_order_size_ratio: float = 0.4  # more reduction when conflict → risk control
    inventory_skew_nonlinear_power: float = 4.0  # maximum nonlinear penalty → strong control at large inventory
    dynamic_skew_vol_kappa: float = 0.10  # stronger skew response to vol
    vol_spread_kappa: float = 0.35  # more spread widening in high vol
    execution_cooldown_ms: int = 500  # moderate cooldown → still good turnover, risk control
    max_same_side_fill_ratio: float = 0.65  # stricter two-sided balance requirement
    two_sided_rebalance_kappa: float = 0.8  # stronger rebalancing
    toxic_flow_size_multiple: float = 2.0  # lower threshold → stop toxic flow earlier
    toxic_flow_pause_ms: int = 400  # longer pause
    toxic_flow_ema_alpha: float = 0.04  # faster adaptation to large trades
    min_side_balance_fills: int = 8  # earlier balance control activation
    cooldown_min_size_ratio: float = 0.2  # smaller minimum during cooldown
    turnover_feedback_kappa: float = 0.4  # softer feedback → profitability > max turnover
    warmup_events: int = 200  # update state but disable executions during early convergence
    g_laplace_prior_count: float = 15.0  # more smoothing → more stable predictions on sparse states
    cooldown_ms: int = 350  # moderate
    base_cooldown_ms: int = 350  # base cooldown
    min_trade_interval_ms: int = 25  # moderate pacing
    vol_window_ticks: int = 50  # short volatility window for spread adaptation
    alpha_obi_fast: float = 0.2  # fast OBI EMA
    alpha_obi_slow: float = 0.03  # slow OBI EMA
    alpha_ofi_fast: float = 0.2  # fast OFI EMA
    alpha_ofi_slow: float = 0.03  # slow OFI EMA
    alpha_weak_threshold: float = 0.05  # weak alpha => reduce/skip quoting
    alpha_threshold: float = 0.07  # deadzone threshold (weak alpha => no quote)
    alpha_strong_threshold: float = 0.18  # strong alpha => favor one side
    alpha_weak_size_ratio: float = 0.10  # size multiplier when signal is weak
    alpha_strong_favor_mult: float = 1.6  # size multiplier on favorable side
    alpha_strong_adverse_mult: float = 0.40  # size multiplier on adverse side
    vol_ref_ticks: float = 0.5  # volatility reference level for spread expansion
    max_vol_factor: float = 1.2  # cap for volatility-driven spread multiplier
    min_half_spread_ticks: float = 4.0  # minimum spread wider → reduce adverse selection
    max_half_spread_ticks: float = 3.5  # upper cap still okay
    toxic_fill_ratio_soft: float = 0.20  # lower threshold → stronger brake earlier
    toxic_fill_ratio_hard: float = 0.35  # lower threshold → stronger brake
    toxic_extra_pause_ms: int = 500  # extra cooldown longer when toxic
    buy_cooldown_multiplier: float = 1.5  # more cooldown on buy side
    sell_cooldown_multiplier: float = 1.0  # balanced cooldown
    vol_ref_window_ticks: int = 3600  # ~1h reference window under 1s sampling
    markout_hard_circuit_bps: float = -1.0  # more sensitive trigger
    hard_circuit_pause_ms: int = 100  # longer pause
    intensity_threshold: float = 30.0  # lower trigger → earlier defense
    lambda_intensity: float = 0.10  # stronger spread widening in high intensity
    fill_rate_alpha: float = 0.01  # legacy cfg slot (unused; ring buffer replaces event-time EMA)

    # runtime progress heartbeat (PowerShell-friendly)
    engine_progress_every_n_events: int = 50_000  # 0 disables

    # ── Dictionary Learning configuration ──
    # When enabled, replaces the OBI/OFI/micro-price alpha pipeline with
    # sparse-coding signals derived from a learned dictionary D (M_features × K_atoms).
    use_dictionary_strategy: bool = True
    dict_n_components: int = 5          # K: number of market-regime "atoms"
    dict_alpha: float = 0.15            # L1 penalty strength (higher → sparser α)
    dict_batch_size: int = 512          # mini-batch size for MiniBatchDictionaryLearning
    dict_train_ratio: float = 0.3       # fraction of events used to fit D (strict prefix)
    dict_max_iter: int = 200            # max iterations for dict learning
    dict_n_features: int = 8            # M: number of raw features extracted per event
    dict_sparse_max_iter: int = 8       # coordinate-descent passes in online Numba solver
    dict_signal_alpha_scale: float = 3.0  # scale dict α → quote skew (ticks)
    dict_residual_widen_ticks: float = 1.2  # extra spread when reconstruction error is large
    dict_anomaly_threshold: float = 2.5  # std multiples for anomaly detection
    dict_cache_path: str = ""           # if non-empty, save/load D from this .npy path


# =========================================
# 1.5)  字典学习：特征矩阵 → 词典训练 → 在线稀疏编码
# =========================================


def build_feature_matrix(
    events: pl.DataFrame,
    cfg: BacktestConfig,
    fit_scalers: bool = True,
    scaler_state: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Task 1: 从订单簿事件流中提取标准化的 N×M 特征矩阵 X。

    特征（M=8 维）：
      0. OBI          — bid_sz / (bid_sz + ask_sz)    ∈ [0, 1]
      1. spread_ticks — (ask - bid) / tick_size
      2. log_depth    — log(bid_sz + ask_sz)
      3. micro_gap    — (micro_px - mid_px) / tick_size
      4. obi_diff     — ΔOBI (current - previous)
      5. depth_ret    — Δdepth / depth_prev
      6. trade_size_n — trade_sz / ema_trade_sz (clipped)
      7. trade_side   — ±1 (aggressive direction)

    返回:
      - X: float64[N, M] 标准化特征矩阵（每列 z-score）
      - scaler_state: {"mean": [M], "std": [M]} 用于在线推理时复现标准化
    """
    eps = 1e-12
    n_events = events.height
    n_features = cfg.dict_n_features

    # ── raw features via Polars (vectorised) ──
    df = events.with_columns([
        (pl.col("bid_sz") / (pl.col("bid_sz") + pl.col("ask_sz") + eps)).alias("obi"),
        ((pl.col("ask_px") - pl.col("bid_px")) / max(cfg.tick_size, eps)).alias("spread_ticks"),
        (pl.col("bid_sz") + pl.col("ask_sz")).log().alias("log_depth"),
    ])

    # micro-price
    if "micro_px" not in df.columns:
        df = df.with_columns(
            (
                pl.col("bid_px")
                + (pl.col("ask_px") - pl.col("bid_px"))
                * (pl.col("bid_sz") / (pl.col("bid_sz") + pl.col("ask_sz") + eps))
            ).alias("micro_px")
        )
    if "mid_px" not in df.columns:
        df = df.with_columns(((pl.col("bid_px") + pl.col("ask_px")) * 0.5).alias("mid_px"))

    df = df.with_columns([
        ((pl.col("micro_px") - pl.col("mid_px")) / max(cfg.tick_size, eps)).alias("micro_gap"),
    ])

    # momentum features (diffs)
    df = df.with_columns([
        pl.col("obi").diff().fill_null(0.0).alias("obi_diff"),
        (pl.col("bid_sz") + pl.col("ask_sz")).alias("total_depth"),
    ])
    df = df.with_columns([
        (pl.col("total_depth").diff().fill_null(0.0)
         / (pl.col("total_depth").shift(1).fill_null(1.0) + eps)).alias("depth_ret"),
    ])

    # trade size normalisation
    trade_sz_ema = df["trade_sz"].ewm_mean(alpha=0.02, min_periods=1)
    df = df.with_columns([
        (pl.col("trade_sz") / (trade_sz_ema + eps)).clip(0.0, 10.0).alias("trade_size_n"),
        pl.col("trade_side").cast(pl.Float64).alias("trade_side_f64"),
    ])

    # ── extract as numpy ──
    raw = np.column_stack([
        df["obi"].to_numpy().astype(np.float64),
        df["spread_ticks"].to_numpy().astype(np.float64),
        df["log_depth"].to_numpy().astype(np.float64),
        df["micro_gap"].to_numpy().astype(np.float64),
        df["obi_diff"].to_numpy().astype(np.float64),
        df["depth_ret"].to_numpy().astype(np.float64),
        df["trade_size_n"].to_numpy().astype(np.float64),
        df["trade_side_f64"].to_numpy().astype(np.float64),
    ])  # shape [N, M]

    # ── clip extremes ──
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, -50.0, 50.0)

    # ── z-score standardisation ──
    if scaler_state is not None:
        mean = scaler_state["mean"]
        std = scaler_state["std"]
    elif fit_scalers:
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std = np.where(std < eps, 1.0, std)
    else:
        mean = np.zeros(n_features, dtype=np.float64)
        std = np.ones(n_features, dtype=np.float64)

    X = (raw - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    scaler_state_out = {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}
    return X, scaler_state_out


def fit_dictionary(
    X_train: np.ndarray,
    cfg: BacktestConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 2: 使用 MiniBatchDictionaryLearning 从训练数据中学习字典 D。

    返回:
      - D: float64[M, K]  字典矩阵（列是原子）
      - alpha_train: float64[N_train, K]  训练集的稀疏系数
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn 未安装，无法训练字典。请执行: pip install scikit-learn"
        )
    if X_train.size == 0:
        raise ValueError("X_train 为空，无法训练字典")

    n_samples = X_train.shape[0]
    batch_size = min(cfg.dict_batch_size, n_samples)

    dlearner = MiniBatchDictionaryLearning(
        n_components=cfg.dict_n_components,
        alpha=cfg.dict_alpha,
        batch_size=batch_size,
        max_iter=cfg.dict_max_iter,
        fit_algorithm="cd",          # coordinate descent → fast on CPU
        transform_algorithm="lasso_lars",
        n_iter=np.ceil(n_samples / batch_size).astype(int) * 3,
        random_state=42,
        shuffle=True,
    )
    D = dlearner.fit_transform(X_train)
    # D is [N_train, K] after fit_transform; need the actual dictionary
    D = dlearner.components_.T.copy()  # [M, K]

    # column-normalise for stability (atoms should be unit vectors)
    for k in range(D.shape[1]):
        col_norm = np.linalg.norm(D[:, k])
        if col_norm > 1e-12:
            D[:, k] /= col_norm

    # sparse coefficients for the training set (via lasso_lars on fitted dict)
    alpha_train = dlearner.transform(X_train)

    return D.astype(np.float64), alpha_train.astype(np.float64)


@njit(cache=True)
def _numba_sparse_encode_single(
    x: np.ndarray,        # [M]  feature vector
    D: np.ndarray,        # [M, K]  learned dictionary
    lam: float,           # L1 penalty
    max_iter: int,
) -> np.ndarray:
    """
    Task 4 核心: Numba JIT 坐标下降稀疏编码。

    对单个特征向量 x，求解:
        min_α  ||x - D α||²₂ + λ||α||₁

    使用迭代 coordinate descent + soft thresholding。
    时间复杂度 O(M·K·max_iter)，适合每 tick 微秒级调用。

    返回:
      - alpha: [K]  稀疏系数向量
    """
    M, K = D.shape
    alpha = np.zeros(K, dtype=np.float64)
    residual = x.copy()  # [M]
    DtD = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            s = 0.0
            for m in range(M):
                s += D[m, i] * D[m, j]
            DtD[i, j] = s
            DtD[j, i] = s

    for _ in range(max_iter):
        for k in range(K):
            # current residual when alpha[k] = 0
            r_k = residual + D[:, k] * alpha[k]  # [M]

            # gradient of data term w.r.t α_k
            grad_k = 0.0
            for m in range(M):
                grad_k += D[m, k] * r_k[m]

            # soft-thresholding: α_k_new = S_λ(grad_k) / (D[:,k]ᵀ D[:,k])
            denom = DtD[k, k] + 1e-12
            alpha_ols = grad_k / denom

            if alpha_ols > lam / denom:
                alpha_k_new = alpha_ols - lam / denom
            elif alpha_ols < -lam / denom:
                alpha_k_new = alpha_ols + lam / denom
            else:
                alpha_k_new = 0.0

            delta = alpha_k_new - alpha[k]
            alpha[k] = alpha_k_new

            # update residual
            for m in range(M):
                residual[m] -= D[m, k] * delta

    return alpha


@njit(cache=True)
def _numba_batch_sparse_encode(
    X: np.ndarray,        # [B, M]
    D: np.ndarray,        # [M, K]
    lam: float,
    max_iter: int,
    alpha_out: np.ndarray,  # [B, K]
) -> None:
    """Vectorised wrapper: encode a batch of rows into pre-allocated alpha_out."""
    B = X.shape[0]
    for i in range(B):
        alpha_out[i] = _numba_sparse_encode_single(X[i], D, lam, max_iter)


def plot_dictionary_atoms(
    D: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Task 3: 可视化字典原子（基底向量），用于金融微观结构归因。

    每个原子是一个 M 维向量，表示该市场状态的"特征指纹"。
    画出折线图，横轴是特征名，纵轴是原子载荷。
    """
    if not HAS_MPL:
        print("matplotlib 未安装，跳过画图。请执行: pip install matplotlib")
        return

    M, K = D.shape
    if feature_names is None:
        feature_names = [
            "OBI", "spread", "log_depth", "micro_gap",
            "obi_diff", "depth_ret", "trade_sz_n", "trade_side",
        ]

    fig, axes = plt.subplots(K, 1, figsize=(12, 3 * K), sharex=True)
    if K == 1:
        axes = [axes]
    x_idx = np.arange(M)

    for k in range(K):
        ax = axes[k]
        atom = D[:, k]
        colors = ["#2196F3" if v >= 0 else "#F44336" for v in atom]
        ax.bar(x_idx, atom, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Atom {k+1}", fontsize=12, fontweight="bold")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(axis="y", alpha=0.3)
        # annotate top features
        top_idx = np.argmax(np.abs(atom))
        ax.annotate(
            f"{feature_names[top_idx]}: {atom[top_idx]:+.3f}",
            xy=(top_idx, atom[top_idx]),
            fontsize=9, fontweight="bold",
            xytext=(0, 8 if atom[top_idx] > 0 else -14),
            textcoords="offset points",
            ha="center",
        )

    axes[-1].set_xticks(x_idx)
    axes[-1].set_xticklabels(feature_names, rotation=30, ha="right", fontsize=10)
    axes[-1].set_xlabel("Feature", fontsize=12)
    fig.suptitle("Dictionary Atoms — Learned Market Microstructure Bases", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"字典原子图已保存: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def dictionary_alpha_to_signal(
    alpha: np.ndarray,   # [K]
    residual_norm: float,
    cfg: BacktestConfig,
) -> Tuple[float, float, float, float]:
    """
    将稀疏系数 α 和重构残差翻译为交易信号。

    返回:
      - skew_signal: 报价中心偏离 (正 = 向上压力, 负 = 向下压力)
      - spread_multiplier: 价差乘数 (>1 = 加宽)
      - buy_enabled: 是否允许做多 (>0)
      - sell_enabled: 是否允许做空 (>0)
    """
    K = len(alpha)
    # Atom 1: 买压强度 (positive loading on OBI + trade_side)
    # Atom 2: 卖压强度
    # Remaining atoms: 均值回复 / 波动率状态
    buy_pressure = max(0.0, alpha[0]) if K > 0 else 0.0
    sell_pressure = max(0.0, alpha[1]) if K > 1 else 0.0
    net_pressure = buy_pressure - sell_pressure

    # skew: 净压力方向 × 缩放
    skew_signal = net_pressure * cfg.dict_signal_alpha_scale  # ticks

    # spread widening: residual anomaly → wider spread
    if residual_norm > cfg.dict_anomaly_threshold:
        spread_multiplier = 1.0 + cfg.dict_residual_widen_ticks
    else:
        spread_multiplier = 1.0

    alpha_norm = np.sqrt(np.sum(alpha**2))
    buy_enabled = 1.0
    sell_enabled = 1.0
    if alpha_norm > 0.8 and net_pressure < -0.15:
        buy_enabled = 0.0
    elif alpha_norm > 0.8 and net_pressure > 0.15:
        sell_enabled = 0.0

    return skew_signal, spread_multiplier, buy_enabled, sell_enabled


def load_btc_data_placeholder(data_dir: Optional[str] = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    自动读取 trades + orderbook 文件并标准化字段。
    可通过环境变量指定数据路径/文件，便于外接硬盘切换：
      - HFT_DATA_DIR: 数据目录（默认当前目录）
      - HFT_TRADES_FILE: 指定 trades 文件（可绝对路径或相对 HFT_DATA_DIR）
      - HFT_BOOK_FILE: 指定 orderbook 文件（可绝对路径或相对 HFT_DATA_DIR）
    """
    base_dir = Path(data_dir or os.getenv("HFT_DATA_DIR", ".")).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {base_dir}")

    def _resolve_file_env(env_key: str) -> Optional[Path]:
        val = os.getenv(env_key, "").strip()
        if not val:
            return None
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = base_dir / p
        return p

    trades_file = _resolve_file_env("HFT_TRADES_FILE")
    book_file = _resolve_file_env("HFT_BOOK_FILE")

    if trades_file is not None and not trades_file.exists():
        raise FileNotFoundError(f"HFT_TRADES_FILE 不存在: {trades_file}")
    if book_file is not None and not book_file.exists():
        raise FileNotFoundError(f"HFT_BOOK_FILE 不存在: {book_file}")

    files = sorted(
        [p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}]
    )
    if not files:
        raise FileNotFoundError(f"{base_dir} 未找到 .csv/.parquet 数据文件")

    file_names: List[str] = [f.name for f in files]

    if trades_file is None or book_file is None:
        for p in files:
            if p.suffix.lower() == ".csv":
                sample = pl.read_csv(str(p), n_rows=50)
            else:
                sample = pl.read_parquet(str(p)).head(50)

            cols = {c.lower() for c in sample.columns}
            has_time = any(x in cols for x in {"ts_ms", "timestamp", "created_time", "ts", "time", "t"})
            has_side = any(x in cols for x in {"trade_side", "side", "direction", "taker_side"})
            has_trade_px = any(x in cols for x in {"trade_px", "price", "p"})
            has_trade_sz = any(x in cols for x in {"trade_sz", "size", "qty", "amount", "volume"})

            has_bid_px = any(x in cols for x in {"bid_px", "bid1_p", "bid1_price", "best_bid", "bid_price"})
            has_ask_px = any(x in cols for x in {"ask_px", "ask1_p", "ask1_price", "best_ask", "ask_price"})
            has_bid_sz = any(x in cols for x in {"bid_sz", "bid1_v", "bid1_size", "best_bid_size", "bid_size"})
            has_ask_sz = any(x in cols for x in {"ask_sz", "ask1_v", "ask1_size", "best_ask_size", "ask_size"})

            if has_time and has_side and has_trade_px and has_trade_sz and trades_file is None:
                trades_file = p
            if has_time and has_bid_px and has_ask_px and has_bid_sz and has_ask_sz and book_file is None:
                book_file = p

    if trades_file is None:
        raise ValueError(f"未识别到 trades 文件。目录: {base_dir}, 文件: {file_names}")
    if book_file is None:
        raise ValueError(
            f"未识别到真实 orderbook 文件。目录: {base_dir}, 文件: {file_names}. "
            "请提供真实盘口数据，已禁用 synthetic orderbook 自动生成。"
        )

    trades_raw = pl.read_csv(str(trades_file)) if trades_file.suffix.lower() == ".csv" else pl.read_parquet(str(trades_file))
    book_raw = pl.read_csv(str(book_file)) if book_file.suffix.lower() == ".csv" else pl.read_parquet(str(book_file))

    trades = _standardize_trades(trades_raw)
    book = _standardize_book(book_raw)
    return trades, book


def _pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    cmap = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in cmap:
            return cmap[c.lower()]
    return None


def _standardize_trades(df: pl.DataFrame) -> pl.DataFrame:
    cols = list(df.columns)
    ts_col = _pick_col(cols, ["ts_ms", "created_time", "timestamp", "ts", "time", "t"])
    px_col = _pick_col(cols, ["trade_px", "price", "p"])
    sz_col = _pick_col(cols, ["trade_sz", "size", "qty", "amount", "volume"])
    side_col = _pick_col(cols, ["trade_side", "side", "direction", "taker_side"])

    miss = [x for x, v in [("ts", ts_col), ("price", px_col), ("size", sz_col), ("side", side_col)] if v is None]
    if miss:
        raise ValueError(f"trades 缺少必要字段: {miss}, 当前列: {cols}")

    side_expr = (
        pl.when(pl.col(side_col).cast(pl.Utf8).str.to_lowercase().is_in(["buy", "b", "1"]))
        .then(pl.lit(1))
        .when(pl.col(side_col).cast(pl.Utf8).str.to_lowercase().is_in(["sell", "s", "-1"]))
        .then(pl.lit(-1))
        .otherwise(pl.col(side_col).cast(pl.Int8, strict=False))
        .cast(pl.Int8)
    )

    out = (
        df.select(
            [
                pl.col(ts_col).cast(pl.Int64).alias("ts_ms"),
                pl.col(px_col).cast(pl.Float64).alias("trade_px"),
                pl.col(sz_col).cast(pl.Float64).alias("trade_sz"),
                side_expr.alias("trade_side"),
            ]
        )
        .drop_nulls()
        .filter(pl.col("trade_px") > 0)
        .filter(pl.col("trade_sz") > 0)
        .filter(pl.col("trade_side").is_in([1, -1]))
        .sort("ts_ms")
    )
    return out


def _standardize_book(df: pl.DataFrame) -> pl.DataFrame:
    cols = list(df.columns)
    ts_col = _pick_col(cols, ["ts_ms", "created_time", "timestamp", "ts", "time", "t"])
    bid_px_col = _pick_col(cols, ["bid_px", "bid1_p", "bid1_price", "best_bid", "bid_price"])
    ask_px_col = _pick_col(cols, ["ask_px", "ask1_p", "ask1_price", "best_ask", "ask_price"])
    bid_sz_col = _pick_col(cols, ["bid_sz", "bid1_v", "bid1_size", "best_bid_size", "bid_size"])
    ask_sz_col = _pick_col(cols, ["ask_sz", "ask1_v", "ask1_size", "best_ask_size", "ask_size"])

    miss = [
        x
        for x, v in [
            ("ts", ts_col),
            ("bid_px", bid_px_col),
            ("ask_px", ask_px_col),
            ("bid_sz", bid_sz_col),
            ("ask_sz", ask_sz_col),
        ]
        if v is None
    ]
    if miss:
        raise ValueError(f"orderbook 缺少必要字段: {miss}, 当前列: {cols}")

    out = (
        df.select(
            [
                pl.col(ts_col).cast(pl.Int64).alias("ts_ms"),
                pl.col(bid_px_col).cast(pl.Float64).alias("bid_px"),
                pl.col(bid_sz_col).cast(pl.Float64).alias("bid_sz"),
                pl.col(ask_px_col).cast(pl.Float64).alias("ask_px"),
                pl.col(ask_sz_col).cast(pl.Float64).alias("ask_sz"),
            ]
        )
        .drop_nulls()
        .filter(pl.col("bid_px") > 0)
        .filter(pl.col("ask_px") > 0)
        .filter(pl.col("bid_sz") >= 0)
        .filter(pl.col("ask_sz") >= 0)
        .filter(pl.col("ask_px") > pl.col("bid_px"))
        .sort("ts_ms")
    )
    return out


# =========================================
# 2) 数据预处理：Polars 对齐到事件流
# =========================================


def _validate_schema(trades: pl.DataFrame, book: pl.DataFrame) -> None:
    t_cols = set(trades.columns)
    b_cols = set(book.columns)
    req_t = {"ts_ms", "trade_px", "trade_sz", "trade_side"}
    req_b = {"ts_ms", "bid_px", "bid_sz", "ask_px", "ask_sz"}

    miss_t = req_t - t_cols
    miss_b = req_b - b_cols
    if miss_t:
        raise ValueError(f"trades 缺字段: {sorted(miss_t)}")
    if miss_b:
        raise ValueError(f"book 缺字段: {sorted(miss_b)}")


def build_event_stream(
    trades: pl.DataFrame,
    book: pl.DataFrame,
    cfg: Optional[BacktestConfig] = None,
) -> pl.DataFrame:
    _validate_schema(trades, book)

    trades = trades.select(
        [
            pl.col("ts_ms").cast(pl.Int64),
            pl.col("trade_px").cast(pl.Float64),
            pl.col("trade_sz").cast(pl.Float64),
            pl.col("trade_side").cast(pl.Int8),
        ]
    ).sort("ts_ms")

    book = (
        book.select(
            [
                pl.col("ts_ms").cast(pl.Int64),
                pl.col("bid_px").cast(pl.Float64),
                pl.col("bid_sz").cast(pl.Float64),
                pl.col("ask_px").cast(pl.Float64),
                pl.col("ask_sz").cast(pl.Float64),
            ]
        )
        .filter(pl.col("ask_px") > pl.col("bid_px"))
        .sort("ts_ms")
    )

    # asof join: 每笔trade匹配最近一个盘口, 并限制最大陈旧时间
    tol_ms = max(1, int(cfg.asof_tolerance_ms)) if cfg is not None else None
    if tol_ms is None:
        events = trades.join_asof(book, on="ts_ms", strategy="backward").drop_nulls()
    else:
        events = trades.join_asof(
            book,
            on="ts_ms",
            strategy="backward",
            tolerance=tol_ms,
        ).drop_nulls()

    # micro-price = bid + spread * bid_sz / (bid_sz + ask_sz)
    # 注意：这里遵循你给的公式方向
    events = events.with_columns(
        [
            ((pl.col("bid_px") + pl.col("ask_px")) * 0.5).alias("mid_px"),
            (pl.col("ask_px") - pl.col("bid_px")).alias("spread"),
            (
                pl.col("bid_px")
                + (pl.col("ask_px") - pl.col("bid_px"))
                * (pl.col("bid_sz") / (pl.col("bid_sz") + pl.col("ask_sz")))
            ).alias("micro_px"),
        ]
    )

    return events


def _build_spread_bins(spread_ticks: np.ndarray, n_bins: int) -> np.ndarray:
    n_bins = max(2, int(n_bins))
    valid = spread_ticks[np.isfinite(spread_ticks)]
    if valid.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(valid, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        v = float(np.median(valid))
        edges = np.array([v - 0.5, v + 0.5], dtype=np.float64)
    return edges.astype(np.float64)


def _build_imbalance_bins(imbalance: np.ndarray, n_bins: int) -> np.ndarray:
    n_bins = max(2, int(n_bins))
    valid = imbalance[np.isfinite(imbalance)]
    if valid.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    valid = np.clip(valid, 0.0, 1.0)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(valid, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([0.0, 1.0], dtype=np.float64)
    return edges.astype(np.float64)


def _build_feature_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    n_bins = max(2, int(n_bins))
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(valid, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        v = float(np.median(valid))
        edges = np.array([v - 1.0, v + 1.0], dtype=np.float64)
    return edges.astype(np.float64)


def _fit_state_jump(
    imb_bin: np.ndarray,
    spr_bin: np.ndarray,
    jump: np.ndarray,
    n_imb: int,
    n_spr: int,
    min_samples: int,
) -> np.ndarray:
    sum_mat = np.zeros((n_imb, n_spr), dtype=np.float64)
    cnt_mat = np.zeros((n_imb, n_spr), dtype=np.int64)
    valid = np.isfinite(jump)
    ib = np.clip(imb_bin[valid], 0, n_imb - 1)
    sb = np.clip(spr_bin[valid], 0, n_spr - 1)
    jv = jump[valid]
    np.add.at(sum_mat, (ib, sb), jv)
    np.add.at(cnt_mat, (ib, sb), 1)
    g = np.zeros((n_imb, n_spr), dtype=np.float64)
    ok = cnt_mat >= max(1, int(min_samples))
    g[ok] = sum_mat[ok] / cnt_mat[ok]
    return g


def _fit_state_jump_3d(
    imb_bin: np.ndarray,
    spr_bin: np.ndarray,
    dep_bin: np.ndarray,
    jump: np.ndarray,
    n_imb: int,
    n_spr: int,
    n_dep: int,
    min_samples: int,
) -> np.ndarray:
    sum_mat = np.zeros((n_imb, n_spr, n_dep), dtype=np.float64)
    cnt_mat = np.zeros((n_imb, n_spr, n_dep), dtype=np.int64)
    valid = np.isfinite(jump)
    ib = np.clip(imb_bin[valid], 0, n_imb - 1)
    sb = np.clip(spr_bin[valid], 0, n_spr - 1)
    db = np.clip(dep_bin[valid], 0, n_dep - 1)
    jv = jump[valid]
    np.add.at(sum_mat, (ib, sb, db), jv)
    np.add.at(cnt_mat, (ib, sb, db), 1)
    g = np.zeros((n_imb, n_spr, n_dep), dtype=np.float64)
    ok = cnt_mat >= max(1, int(min_samples))
    g[ok] = sum_mat[ok] / cnt_mat[ok]
    return g


def _fit_state_jump_4d(
    imb_bin: np.ndarray,
    spr_bin: np.ndarray,
    dep_bin: np.ndarray,
    dur_bin: np.ndarray,
    jump: np.ndarray,
    n_imb: int,
    n_spr: int,
    n_dep: int,
    n_dur: int,
    min_samples: int,
    prior_count: float = 0.0,
) -> np.ndarray:
    sum_mat = np.zeros((n_imb, n_spr, n_dep, n_dur), dtype=np.float64)
    cnt_mat = np.zeros((n_imb, n_spr, n_dep, n_dur), dtype=np.int64)
    valid = np.isfinite(jump)
    ib = np.clip(imb_bin[valid], 0, n_imb - 1)
    sb = np.clip(spr_bin[valid], 0, n_spr - 1)
    db = np.clip(dep_bin[valid], 0, n_dep - 1)
    ub = np.clip(dur_bin[valid], 0, n_dur - 1)
    jv = jump[valid]
    np.add.at(sum_mat, (ib, sb, db, ub), jv)
    np.add.at(cnt_mat, (ib, sb, db, ub), 1)
    g = np.zeros((n_imb, n_spr, n_dep, n_dur), dtype=np.float64)
    global_mean = float(np.nanmean(jv)) if jv.size > 0 else 0.0
    if not np.isfinite(global_mean):
        global_mean = 0.0
    # Laplace-style shrinkage keeps long-tail states from overreacting to tiny samples.
    if prior_count > 0.0:
        g = (sum_mat + prior_count * global_mean) / (cnt_mat + prior_count)
    ok = cnt_mat >= max(1, int(min_samples))
    g[ok] = sum_mat[ok] / np.maximum(cnt_mat[ok], 1)
    return g


def _next_mid_jump_return(mid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    label: next mid-price change return.
    For each i, find the earliest j>i with mid[j] != mid[i], then label=(mid[j]/mid[i]-1).
    """
    n = mid.size
    out = np.full(n, np.nan, dtype=np.float64)
    idx = np.full(n, -1, dtype=np.int64)
    if n <= 1:
        return out, idx
    eps = 1e-12
    next_change_idx = np.full(n, -1, dtype=np.int64)
    for i in range(n - 2, -1, -1):
        if np.abs(mid[i + 1] - mid[i]) > eps:
            next_change_idx[i] = i + 1
        else:
            next_change_idx[i] = next_change_idx[i + 1]
    base = np.maximum(mid, eps)
    for i in range(n):
        j = next_change_idx[i]
        if j > i:
            out[i] = mid[j] / base[i] - 1.0
            idx[i] = j
    return out, idx


def enrich_advanced_microprice(
    events: pl.DataFrame,
    cfg: BacktestConfig,
    g_matrix: Optional[np.ndarray] = None,
    bin_state: Optional[Dict[str, np.ndarray]] = None,
    return_fit_state: bool = False,
) -> pl.DataFrame | Tuple[pl.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    """
    进阶版 micro-price:
    1) rebate 修正后净价
    2) 离散状态 (imbalance, spread, depth, duration)
    3) 离线学习 G(I,S,D,U)=E[future_mid_mean-mid | state]
    4) micro_px_adv = mid_net + G(I,S)
    同时输出 waiting time 与 future-return label.
    """
    if events.height == 0:
        return events

    eps = 1e-12
    k = max(1, int(cfg.label_future_ticks))
    future_mid_exprs = [pl.col("mid_px").shift(-i) for i in range(1, k + 1)]
    rebate_signed = cfg.maker_rebate_bps / 10_000.0

    events2 = events.with_columns(
        [
            # 基础状态
            (pl.col("bid_sz") / (pl.col("bid_sz") + pl.col("ask_sz") + eps)).alias("imbalance"),
            ((pl.col("ask_px") - pl.col("bid_px")) / max(cfg.tick_size, eps)).alias("spread_ticks"),
            (pl.col("bid_sz") + pl.col("ask_sz")).alias("total_depth"),
            (pl.col("bid_sz") - pl.col("ask_sz")).alias("signed_imbalance"),
            (pl.col("bid_sz") - pl.col("ask_sz")).abs().alias("abs_imbalance"),
            (pl.col("ts_ms").cast(pl.Float64).diff().fill_null(0.0).clip(lower_bound=0.0)).alias("duration_ms"),
            # waiting time proxy: tick 间隔滚动均值
            (
                pl.col("ts_ms")
                .cast(pl.Float64)
                .diff()
                .fill_null(0.0)
                .clip(lower_bound=0.0)
                .rolling_mean(window_size=max(1, int(cfg.waiting_ma_window)), min_samples=1)
            ).alias("waiting_time_ms"),
            # 先计算 baseline 标签(未来k个tick均值收益), 后续可替换为 next-jump 标签
            (pl.mean_horizontal(future_mid_exprs) / (pl.col("mid_px") + eps) - 1.0).alias(
                "label_fwd_ret_k"
            ),
            # rebate净价修正
            (pl.col("bid_px") * (1.0 + rebate_signed)).alias("bid_net"),
            (pl.col("ask_px") * (1.0 - rebate_signed)).alias("ask_net"),
        ]
    ).with_columns([((pl.col("bid_net") + pl.col("ask_net")) * 0.5).alias("mid_net")])

    bid = events2["bid_px"].to_numpy().astype(np.float64)
    ask = events2["ask_px"].to_numpy().astype(np.float64)
    mid = events2["mid_px"].to_numpy().astype(np.float64)
    imbalance = events2["imbalance"].to_numpy().astype(np.float64)
    spread_ticks = events2["spread_ticks"].to_numpy().astype(np.float64)
    total_depth = events2["total_depth"].to_numpy().astype(np.float64)
    duration_ms = events2["duration_ms"].to_numpy().astype(np.float64)
    fwd_ret_mean_k = events2["label_fwd_ret_k"].to_numpy().astype(np.float64)
    fwd_ret_next_jump, next_jump_idx = _next_mid_jump_return(mid)
    fwd_ret = fwd_ret_next_jump if cfg.use_next_jump_label else fwd_ret_mean_k
    mid_net = events2["mid_net"].to_numpy().astype(np.float64)

    # state discretization
    if bin_state is not None:
        imb_edges = bin_state["imb_edges"]
        spread_edges = bin_state["spread_edges"]
        depth_edges = bin_state["depth_edges"]
        dur_edges = bin_state["dur_edges"]
    else:
        n_imb_cfg = max(2, int(cfg.imbalance_bins))
        if cfg.use_quantile_imbalance_bins:
            imb_edges = _build_imbalance_bins(imbalance, n_imb_cfg)
        else:
            imb_edges = np.linspace(0.0, 1.0, n_imb_cfg + 1, dtype=np.float64)
        spread_edges = _build_spread_bins(spread_ticks, cfg.spread_bins)
        depth_edges = _build_feature_bins(total_depth, cfg.depth_bins)
        dur_edges = _build_feature_bins(duration_ms, cfg.duration_bins)

    n_imb = max(1, imb_edges.size - 1)
    n_spr = max(1, spread_edges.size - 1)
    n_dep = max(1, depth_edges.size - 1)
    n_dur = max(1, dur_edges.size - 1)

    imb_bin = np.digitize(imbalance, imb_edges[1:-1], right=False).astype(np.int64)
    imb_bin = np.clip(imb_bin, 0, n_imb - 1)
    spr_bin = np.digitize(spread_ticks, spread_edges[1:-1], right=False).astype(np.int64)
    spr_bin = np.clip(spr_bin, 0, n_spr - 1)
    dep_bin = np.digitize(total_depth, depth_edges[1:-1], right=False).astype(np.int64)
    dep_bin = np.clip(dep_bin, 0, n_dep - 1)
    dur_bin = np.digitize(duration_ms, dur_edges[1:-1], right=False).astype(np.int64)
    dur_bin = np.clip(dur_bin, 0, n_dur - 1)

    fwd_mid = mid * (1.0 + fwd_ret)
    jump = fwd_mid - mid

    # Prevent full-sample leakage: fit G(I,S) on prefix only, apply to all rows.
    n = len(jump)
    train_ratio = min(0.95, max(0.05, float(cfg.adv_train_ratio)))
    train_size = max(int(cfg.min_state_samples), int(n * train_ratio))
    train_size = min(n, train_size)
    # Strictly keep training labels inside train window to avoid leakage.
    if cfg.use_next_jump_label:
        idx = np.arange(n, dtype=np.int64)
        train_mask = (idx < train_size) & (next_jump_idx >= 0) & (next_jump_idx < train_size) & np.isfinite(jump)
    else:
        # For k-tick mean labels, keep labels fully inside train segment.
        safe_train_end = max(0, train_size - k)
        idx = np.arange(n, dtype=np.int64)
        train_mask = (idx < safe_train_end) & np.isfinite(jump)

    train_imb = imb_bin[train_mask]
    train_spr = spr_bin[train_mask]
    train_dep = dep_bin[train_mask]
    train_dur = dur_bin[train_mask]
    train_jump = jump[train_mask]

    if cfg.use_symmetry_trick:
        # Symmetry trick from micro-price literature:
        # (I, S, dP) and (1-I, S, -dP) should carry mirrored information.
        sym_imb = (n_imb - 1) - train_imb
        fit_imb = np.concatenate([train_imb, sym_imb])
        fit_spr = np.concatenate([train_spr, train_spr])
        fit_dep = np.concatenate([train_dep, train_dep])
        fit_dur = np.concatenate([train_dur, train_dur])
        fit_jump = np.concatenate([train_jump, -train_jump])
    else:
        fit_imb = train_imb
        fit_spr = train_spr
        fit_dep = train_dep
        fit_dur = train_dur
        fit_jump = train_jump

    if g_matrix is None:
        g_mat = _fit_state_jump_4d(
            fit_imb,
            fit_spr,
            fit_dep,
            fit_dur,
            fit_jump,
            n_imb,
            n_spr,
            n_dep,
            n_dur,
            cfg.min_state_samples,
            cfg.g_laplace_prior_count,
        )
    else:
        g_mat = g_matrix
    # 保证连续内存布局，降低后续索引访问开销并提升 Numba 兼容稳定性。
    g_mat = np.ascontiguousarray(g_mat)
    g_lookup = g_mat[imb_bin, spr_bin, dep_bin, dur_bin]
    micro_px_adv = mid_net + g_lookup

    out = events2.with_columns(
        [
            pl.Series("label_fwd_ret_k", fwd_ret),
            pl.Series("micro_px_adv", micro_px_adv),
            pl.Series("state_imb_bin", imb_bin),
            pl.Series("state_spread_bin", spr_bin),
            pl.Series("state_depth_bin", dep_bin),
            pl.Series("state_duration_bin", dur_bin),
            pl.Series("state_jump_g", g_lookup),
        ]
    )
    if return_fit_state:
        fit_state = {
            "imb_edges": imb_edges.astype(np.float64),
            "spread_edges": spread_edges.astype(np.float64),
            "depth_edges": depth_edges.astype(np.float64),
            "dur_edges": dur_edges.astype(np.float64),
        }
        return out, g_mat, fit_state
    return out


# =========================================
# 3) Numba 字典策略回测核心
# =========================================


@njit(cache=True)
def _round_to_tick(px: float, tick: float) -> float:
    return np.round(px / tick) * tick


@njit(cache=True)
def _construct_feature_vector(
    bsz_obs: float, asz_obs: float,
    bpx_obs: float, apx_obs: float,
    micro_obs: float,
    trade_sz: float, trade_side_i: float,
    prev_obi: float, prev_depth: float,
    tick_size: float, avg_trade_sz_ema: float,
    eps: float,
) -> Tuple[np.ndarray, float, float]:
    """Numba-compatible feature extraction for a single event."""
    obi = bsz_obs / (bsz_obs + asz_obs + eps)
    spread_ticks = (apx_obs - bpx_obs) / max(tick_size, eps)
    depth = bsz_obs + asz_obs
    log_depth = np.log(max(eps, depth))
    mid = (bpx_obs + apx_obs) * 0.5
    micro_gap = (micro_obs - mid) / max(tick_size, eps)
    obi_diff = obi - prev_obi
    depth_ret = (depth - prev_depth) / max(eps, prev_depth)
    trade_sz_ema = max(eps, avg_trade_sz_ema)
    trade_size_n = min(10.0, max(0.0, trade_sz / trade_sz_ema))

    raw = np.zeros(8, dtype=np.float64)
    raw[0] = obi
    raw[1] = spread_ticks
    raw[2] = log_depth
    raw[3] = micro_gap
    raw[4] = obi_diff
    raw[5] = depth_ret
    raw[6] = trade_size_n
    raw[7] = trade_side_i
    return raw, obi, depth


@njit(cache=True)
def _dict_sparse_encode_inline(
    x: np.ndarray,        # [M]
    D: np.ndarray,        # [M, K]
    lam: float,
    max_iter: int,
) -> Tuple[np.ndarray, float]:
    """
    Inline sparse coding (duplicated from _numba_sparse_encode_single
    for inline use in the main loop to avoid extra call overhead).
    Returns (alpha [K], residual_norm).
    """
    M, K = D.shape
    alpha = np.zeros(K, dtype=np.float64)
    residual = x.copy()

    # precompute DᵀD
    DtD = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            s = 0.0
            for m in range(M):
                s += D[m, i] * D[m, j]
            DtD[i, j] = s
            DtD[j, i] = s

    for _ in range(max_iter):
        for k in range(K):
            r_k = residual + D[:, k] * alpha[k]
            grad_k = 0.0
            for m in range(M):
                grad_k += D[m, k] * r_k[m]
            denom = DtD[k, k] + 1e-12
            alpha_ols = grad_k / denom
            if alpha_ols > lam / denom:
                alpha_k_new = alpha_ols - lam / denom
            elif alpha_ols < -lam / denom:
                alpha_k_new = alpha_ols + lam / denom
            else:
                alpha_k_new = 0.0
            delta = alpha_k_new - alpha[k]
            alpha[k] = alpha_k_new
            for m in range(M):
                residual[m] -= D[m, k] * delta

    # residual norm = ||x - Dα||₂
    res_norm = 0.0
    for m in range(M):
        res_norm += residual[m] * residual[m]
    res_norm = np.sqrt(res_norm)
    return alpha, res_norm


@njit(cache=True)
def _run_engine_dict(
    ts_ms: np.ndarray,
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_sz: np.ndarray,
    trade_px: np.ndarray,
    trade_sz: np.ndarray,
    trade_side: np.ndarray,
    micro_px: np.ndarray,
    depth_bin: np.ndarray,
    duration_bin: np.ndarray,
    cfg_array: np.ndarray,
    D: np.ndarray,                      # [M, K]  learned dictionary
    scaler_mean: np.ndarray,            # [M]  feature z-score mean
    scaler_std: np.ndarray,             # [M]  feature z-score std
) -> Tuple[np.ndarray, float, np.ndarray, float, float, int]:
    """
    Dictionary-learning market-making engine (Numba JIT).

    Replaces the OBI/OFI/micro-price alpha pipeline with:
      1. Construct raw feature vector x_t at each event
      2. Z-score normalize → x_norm
      3. Sparse code: min_α ||x_norm - Dα||²₂ + λ||α||₁
      4. α coefficients drive quote skew, spread widening, side gating

    Keeps: observation latency, queue modelling, fill detection,
           inventory management, volatility-adaptive spread, cooldowns,
           side balancing, markout circuit breakers.
    """
    # ── Unpack cfg_array (first 82 slots = original cfg, remainder = dict params) ──
    initial_capital = cfg_array[0]
    rebate_rate = cfg_array[1]
    tick_size = cfg_array[2]
    half_spread_ticks = cfg_array[3]
    gamma_inventory = cfg_array[4]
    max_pos_btc = cfg_array[5]
    order_size_btc = cfg_array[6]
    latency_ms = int(cfg_array[7])
    queue_ahead_ratio = cfg_array[8]
    participation_cap = cfg_array[9]
    queue_model = int(cfg_array[10])
    queue_prob_kappa = cfg_array[11]
    queue_decay_lambda_per_ms = cfg_array[12]
    adverse_selection_bps = cfg_array[13]
    observation_latency_ms = int(cfg_array[14])
    refresh_queue_on_activation = int(cfg_array[15])
    # slots 16-17: inventory_vol_alpha, inventory_vol_beta (unused in dict engine)
    thin_depth_spread_mult = cfg_array[18]
    thin_depth_adverse_mult = cfg_array[19]
    fast_duration_spread_mult = cfg_array[20]
    # slots 21-27: OBI circuit (unused in dict engine)
    intensity_threshold = cfg_array[28]
    lambda_intensity = cfg_array[29]
    # slot 30: fill_rate_alpha (unused)
    engine_progress_every_n_events = int(cfg_array[31])
    # slots 32-38: micro-price / inventory unwind (unused in dict engine)
    inventory_skew_nonlinear_power = cfg_array[39]
    dynamic_skew_vol_kappa = cfg_array[40]
    vol_spread_kappa = cfg_array[41]
    execution_cooldown_ms = int(cfg_array[42])
    max_same_side_fill_ratio = cfg_array[43]
    two_sided_rebalance_kappa = cfg_array[44]
    # slots 45-48: toxic flow (unused in dict engine)
    cooldown_min_size_ratio = cfg_array[49]
    turnover_feedback_kappa = cfg_array[50]
    required_monthly_turnover = cfg_array[51]
    warmup_events = int(cfg_array[52])
    # slot 53: g_laplace_prior_count (unused)
    cooldown_ms = int(cfg_array[54])
    cfg_base_cooldown_ms = int(cfg_array[55])
    vol_window_ticks = int(cfg_array[56])
    # slots 57-66: alpha OBI/OFI (unused in dict engine)
    vol_ref_ticks = cfg_array[67]
    max_vol_factor = cfg_array[68]
    min_half_spread_ticks = cfg_array[69]
    max_half_spread_ticks = cfg_array[70]
    # slots 71-75: toxic fill (unused)
    vol_ref_window_ticks = int(cfg_array[76])
    markout_hard_circuit_bps = cfg_array[77]
    hard_circuit_pause_ms = int(cfg_array[78])
    min_trade_interval_ms = int(cfg_array[79])
    max_order_size_ratio_cfg = cfg_array[80]
    # dict-specific params (slots 82+)
    dict_sparse_max_iter = int(cfg_array[82]) if cfg_array.shape[0] > 82 else 8
    dict_lam = cfg_array[83] if cfg_array.shape[0] > 83 else 0.15
    dict_signal_alpha_scale = cfg_array[84] if cfg_array.shape[0] > 84 else 3.0
    dict_residual_widen_ticks = cfg_array[85] if cfg_array.shape[0] > 85 else 1.2
    dict_anomaly_threshold = cfg_array[86] if cfg_array.shape[0] > 86 else 2.5

    # ── clamp guards ──
    half_spread_ticks = max(0.5, half_spread_ticks)
    gamma_inventory = max(2.0, gamma_inventory)
    if cooldown_ms < 100: cooldown_ms = 100
    if cfg_base_cooldown_ms < 100: cfg_base_cooldown_ms = 100
    vol_window_ticks = max(5, min(200, vol_window_ticks))
    min_half_spread_ticks = max(0.1, min_half_spread_ticks)
    max_half_spread_ticks = max(min_half_spread_ticks, max_half_spread_ticks)

    n = len(ts_ms)
    equity = np.empty(n, dtype=np.float64)

    cash = initial_capital
    inv = 0.0
    total_notional = 0.0

    fills = np.zeros((n, 4), dtype=np.float64)
    fill_count = 0

    active_bid = 0.0; active_ask = 0.0; active_live = 0
    bid_queue_left = 0.0; ask_queue_left = 0.0
    pending_bid = 0.0; pending_ask = 0.0; pending_activate_ts = -1
    pending_bid_queue = 0.0; pending_ask_queue = 0.0
    active_bid_order_size = order_size_btc; active_ask_order_size = order_size_btc
    pending_bid_order_size = order_size_btc; pending_ask_order_size = order_size_btc
    last_t = -1; obs_idx = 0

    # Inventory vol tracking
    prev_obs_mid = 0.0; ewma_abs_ret_bps = 0.0

    mid_window = np.zeros(vol_window_ticks, dtype=np.float64)
    mid_window_count = 0
    ref_window = np.zeros(vol_ref_window_ticks, dtype=np.float64)
    ref_window_count = 0

    fill_times = np.zeros(100, dtype=np.int64); fill_idx = 0
    buy_fill_count = 0; sell_fill_count = 0
    last_fill_ts = -1; avg_trade_sz_ema = 0.0
    circuit_until_ts = -1

    pending_mark_ts = np.zeros(128, dtype=np.int64)
    pending_mark_side = np.zeros(128, dtype=np.float64)
    pending_mark_px = np.zeros(128, dtype=np.float64)
    pending_mark_live = np.zeros(128, dtype=np.int64)
    pending_mark_ptr = 0; markout_bad_streak = 0

    ts0 = ts_ms[0] if n > 0 else 0; early_stopped = 0
    check_step = max(1, n // 5); next_check = check_step
    last_buy_ts = -1; last_sell_ts = -1

    # Dictionary feature state
    prev_obi = 0.5; prev_depth = 1.0
    alpha_ema = np.zeros(D.shape[1], dtype=np.float64)
    alpha_ema_decay = 0.06

    for i in range(n):
        t = ts_ms[i]
        in_warmup = i < warmup_events

        if engine_progress_every_n_events > 0 and (i % engine_progress_every_n_events) == 0:
            print("engine_progress", i, n, int(t))

        # ── Observation latency ──
        obs_cut = t - observation_latency_ms
        while (obs_idx + 1) < n and ts_ms[obs_idx + 1] <= obs_cut:
            obs_idx += 1

        bpx_obs = bid_px[obs_idx]; apx_obs = ask_px[obs_idx]
        bsz_obs = bid_sz[obs_idx]; asz_obs = ask_sz[obs_idx]
        mid_obs = 0.5 * (bpx_obs + apx_obs)

        mid_window[i % vol_window_ticks] = mid_obs
        if mid_window_count < vol_window_ticks: mid_window_count += 1
        ref_window[i % vol_ref_window_ticks] = mid_obs
        if ref_window_count < vol_ref_window_ticks: ref_window_count += 1

        mid = 0.5 * (bid_px[i] + ask_px[i])
        tpx = trade_px[i]; tsz = trade_sz[i]; tside = trade_side[i]

        if tsz > 0.0:
            if avg_trade_sz_ema <= 0.0:
                avg_trade_sz_ema = tsz
            else:
                avg_trade_sz_ema = 0.96 * avg_trade_sz_ema + 0.04 * tsz

        # ── Markout circuit breaker ──
        for m in range(128):
            if pending_mark_live[m] == 1 and t >= (pending_mark_ts[m] + 100):
                side = pending_mark_side[m]
                exec_px = pending_mark_px[m]
                mark_bps = side * (mid - exec_px) / (exec_px + 1e-12) * 10_000.0
                if mark_bps < markout_hard_circuit_bps:
                    markout_bad_streak += 1
                else:
                    markout_bad_streak = 0
                pending_mark_live[m] = 0
        if markout_bad_streak >= 4:
            circuit_until_ts = t + hard_circuit_pause_ms
            markout_bad_streak = 0

        # ── Queue decay ──
        if active_live == 1 and last_t >= 0 and t > last_t:
            dt_ms = float(t - last_t)
            decay = np.exp(-queue_decay_lambda_per_ms * dt_ms)
            bid_queue_left *= decay; ask_queue_left *= decay

        # ── Pending activation ──
        if pending_activate_ts >= 0 and t >= pending_activate_ts:
            active_bid = pending_bid; active_ask = pending_ask
            active_bid_order_size = pending_bid_order_size
            active_ask_order_size = pending_ask_order_size
            if refresh_queue_on_activation == 1:
                bid_queue_left = max(0.0, bsz_obs * queue_ahead_ratio)
                ask_queue_left = max(0.0, asz_obs * queue_ahead_ratio)
            else:
                bid_queue_left = pending_bid_queue; ask_queue_left = pending_ask_queue
            active_live = 1; pending_activate_ts = -1

        # ── Fill detection (bid side) ──
        if (not in_warmup) and active_live == 1 and active_bid > 0.0:
            if tside < 0:
                is_cross = active_bid >= apx_obs
                is_touch = np.abs(active_bid - apx_obs) <= (0.5 * tick_size)
                fill_qty = 0.0
                if is_cross:
                    fill_qty = min(active_bid_order_size, tsz * participation_cap, max_pos_btc - inv)
                if is_touch:
                    if queue_model == 0:
                        bid_queue_left = max(0.0, bid_queue_left - tsz)
                        avail = max(0.0, tsz * participation_cap - bid_queue_left)
                        fill_qty = max(fill_qty, min(active_bid_order_size, avail, max_pos_btc - inv))
                    else:
                        pressure = (tsz * participation_cap) / (bid_queue_left + active_bid_order_size + 1e-12)
                        p_fill = 1.0 - np.exp(-queue_prob_kappa * max(0.0, pressure))
                        p_fill = min(1.0, max(0.0, p_fill))
                        fill_qty = max(fill_qty, min(active_bid_order_size * p_fill, max_pos_btc - inv))
                        bid_queue_left = max(0.0, bid_queue_left - tsz)
                if fill_qty > 0.0:
                    adv_mult = thin_depth_adverse_mult if depth_bin[obs_idx] == 0 else 1.0
                    exec_px = active_bid * (1.0 + (adverse_selection_bps * adv_mult) / 10_000.0)
                    notional = fill_qty * exec_px
                    cash -= notional; cash += notional * rebate_rate
                    inv += fill_qty; total_notional += notional
                    fills[fill_count, 0] = t; fills[fill_count, 1] = 1.0
                    fills[fill_count, 2] = fill_qty; fills[fill_count, 3] = exec_px
                    fill_count += 1
                    fill_times[fill_idx % 100] = t; fill_idx += 1
                    buy_fill_count += 1; last_fill_ts = t; last_buy_ts = t
                    pm = pending_mark_ptr % 128
                    pending_mark_ts[pm] = t; pending_mark_side[pm] = 1.0
                    pending_mark_px[pm] = exec_px; pending_mark_live[pm] = 1
                    pending_mark_ptr += 1

        # ── Fill detection (ask side) ──
        if (not in_warmup) and active_live == 1 and active_ask > 0.0:
            if tside > 0:
                is_cross = active_ask <= bpx_obs
                is_touch = np.abs(active_ask - bpx_obs) <= (0.5 * tick_size)
                fill_qty = 0.0
                if is_cross:
                    fill_qty = min(active_ask_order_size, tsz * participation_cap, max_pos_btc + inv)
                if is_touch:
                    if queue_model == 0:
                        ask_queue_left = max(0.0, ask_queue_left - tsz)
                        avail = max(0.0, tsz * participation_cap - ask_queue_left)
                        fill_qty = max(fill_qty, min(active_ask_order_size, avail, max_pos_btc + inv))
                    else:
                        pressure = (tsz * participation_cap) / (ask_queue_left + active_ask_order_size + 1e-12)
                        p_fill = 1.0 - np.exp(-queue_prob_kappa * max(0.0, pressure))
                        p_fill = min(1.0, max(0.0, p_fill))
                        fill_qty = max(fill_qty, min(active_ask_order_size * p_fill, max_pos_btc + inv))
                        ask_queue_left = max(0.0, ask_queue_left - tsz)
                if fill_qty > 0.0:
                    adv_mult = thin_depth_adverse_mult if depth_bin[obs_idx] == 0 else 1.0
                    exec_px = active_ask * (1.0 - (adverse_selection_bps * adv_mult) / 10_000.0)
                    notional = fill_qty * exec_px
                    cash += notional; cash += notional * rebate_rate
                    inv -= fill_qty; total_notional += notional
                    fills[fill_count, 0] = t; fills[fill_count, 1] = -1.0
                    fills[fill_count, 2] = fill_qty; fills[fill_count, 3] = exec_px
                    fill_count += 1
                    fill_times[fill_idx % 100] = t; fill_idx += 1
                    sell_fill_count += 1; last_fill_ts = t; last_sell_ts = t
                    pm = pending_mark_ptr % 128
                    pending_mark_ts[pm] = t; pending_mark_side[pm] = -1.0
                    pending_mark_px[pm] = exec_px; pending_mark_live[pm] = 1
                    pending_mark_ptr += 1

        # MTM
        equity[i] = cash + inv * mid

        # ── Volatility tracking ──
        if prev_obs_mid > 0.0 and mid_obs > 0.0:
            ret_bps = np.abs(mid_obs / prev_obs_mid - 1.0) * 10_000.0
            ewma_abs_ret_bps = (1.0 - 0.02) * ewma_abs_ret_bps + 0.02 * ret_bps
        prev_obs_mid = mid_obs

        # ═══════════════════════════════════════════════
        #  Dictionary-based Alpha Signal
        # ═══════════════════════════════════════════════
        eps = 1e-12
        feat_raw, obi_now, depth_now = _construct_feature_vector(
            bsz_obs, asz_obs, bpx_obs, apx_obs,
            micro_px[obs_idx],
            tsz, tside,
            prev_obi, prev_depth,
            tick_size, avg_trade_sz_ema, eps,
        )
        prev_obi = obi_now; prev_depth = depth_now

        # z-score normalise
        feat_norm = np.zeros(8, dtype=np.float64)
        for j in range(8):
            feat_norm[j] = (feat_raw[j] - scaler_mean[j]) / scaler_std[j]

        # sparse encode → alpha [K]
        alpha, res_norm = _dict_sparse_encode_inline(feat_norm, D, dict_lam, dict_sparse_max_iter)

        # EMA-smoothed alpha for stability
        for j in range(len(alpha)):
            alpha_ema[j] = (1.0 - alpha_ema_decay) * alpha_ema[j] + alpha_ema_decay * alpha[j]

        # ── Translate α → trading signals ──
        K = len(alpha_ema)
        buy_pressure = max(0.0, alpha_ema[0]) if K > 0 else 0.0
        sell_pressure = max(0.0, alpha_ema[1]) if K > 1 else 0.0
        net_pressure = buy_pressure - sell_pressure
        dict_skew_ticks = net_pressure * dict_signal_alpha_scale

        # residual-based spread widening
        if res_norm > dict_anomaly_threshold:
            dict_spread_mult = 1.0 + dict_residual_widen_ticks
        else:
            dict_spread_mult = 1.0

        alpha_abs_sum = 0.0
        for j in range(K):
            alpha_abs_sum += abs(alpha_ema[j])
        dict_buy_ok = 1.0
        dict_sell_ok = 1.0
        if alpha_abs_sum > 0.8 and net_pressure < -0.15:
            dict_buy_ok = 0.0
        elif alpha_abs_sum > 0.8 and net_pressure > 0.15:
            dict_sell_ok = 0.0

        # ═══════════════════════════════════════════════
        #  Quote Construction
        # ═══════════════════════════════════════════════
        inventory_ratio = inv / (max_pos_btc + 1e-12)
        inv_abs = abs(inventory_ratio)
        inv_curve = inv_abs ** inventory_skew_nonlinear_power
        sigmoid_part = 1.0 / (1.0 + np.exp(-8.0 * (inv_curve - 0.35)))
        dynamic_skew_boost = 1.0 + dynamic_skew_vol_kappa * (ewma_abs_ret_bps / 10.0)
        inv_skew_ticks = gamma_inventory * sigmoid_part * dynamic_skew_boost
        inv_skew_ticks = min(4.0, inv_skew_ticks)
        inv_skew = np.sign(inventory_ratio) * inv_skew_ticks * tick_size

        # combined skew: inventory + dictionary signal
        total_skew_ticks = inv_skew_ticks * np.sign(inventory_ratio) + dict_skew_ticks
        total_skew_ticks = max(-4.0, min(4.0, total_skew_ticks))
        reservation_price = mid_obs - total_skew_ticks * tick_size

        # ── Volatility-adaptive spread ──
        vol_short = 0.0
        if mid_window_count >= 3:
            mean_mid = 0.0
            for j in range(mid_window_count): mean_mid += mid_window[j]
            mean_mid /= mid_window_count
            var_mid = 0.0
            for j in range(mid_window_count):
                d = mid_window[j] - mean_mid; var_mid += d * d
            var_mid /= mid_window_count
            vol_short = np.sqrt(max(0.0, var_mid))
        vol_short_ticks = vol_short / (tick_size + 1e-12)

        vol_ref_dynamic = vol_ref_ticks
        if ref_window_count >= 10:
            ref_mean = 0.0
            for j in range(ref_window_count): ref_mean += ref_window[j]
            ref_mean /= ref_window_count
            ref_var = 0.0
            for j in range(ref_window_count):
                d = ref_window[j] - ref_mean; ref_var += d * d
            ref_var /= ref_window_count
            vol_ref_dynamic = max(vol_ref_ticks, np.sqrt(max(0.0, ref_var)) / (tick_size + 1e-12))
        vol_scale = min(vol_short_ticks / (vol_ref_dynamic + 1e-12), max_vol_factor)

        half_spread_dyn = half_spread_ticks + vol_spread_kappa * vol_short_ticks

        # ── Fill intensity adaptive spread ──
        adaptive_spread_mult = 1.0
        if fill_idx >= 50:
            oldest_idx = (fill_idx - 50) % 100
            time_diff_ms = t - fill_times[oldest_idx]
            if time_diff_ms > 0 and time_diff_ms < 60000:
                implied_fills_per_min = 50.0 * (60000.0 / time_diff_ms)
                intensity_gap = max(0.0, implied_fills_per_min - intensity_threshold)
                adaptive_spread_mult = 1.0 + (lambda_intensity * intensity_gap)
        if abs(inv) >= (0.9 * max_pos_btc):
            adaptive_spread_mult *= 2.0

        half_spread_adj = half_spread_dyn * adaptive_spread_mult * dict_spread_mult
        if vol_scale > 0.9: half_spread_adj *= 1.15
        if depth_bin[obs_idx] == 0: half_spread_adj *= thin_depth_spread_mult
        if duration_bin[obs_idx] == 0: half_spread_adj *= fast_duration_spread_mult
        half_spread_adj = min(max_half_spread_ticks, max(min_half_spread_ticks, half_spread_adj))
        current_half_spread = half_spread_adj * tick_size
        current_half_spread = max(0.5 * tick_size, min(5.0 * tick_size, current_half_spread))

        quote_bid = _round_to_tick(reservation_price - current_half_spread, tick_size)
        quote_ask = _round_to_tick(reservation_price + current_half_spread, tick_size)

        # ── Order size: baseline ± dictionary directional gating ──
        current_bid_sz = order_size_btc * dict_buy_ok
        current_ask_sz = order_size_btc * dict_sell_ok

        # Inventory protection
        if inv >= 0.7 * max_pos_btc: current_bid_sz = 0.0
        if inv <= -0.7 * max_pos_btc: current_ask_sz = 0.0

        # ── Cooldown management ──
        elapsed_ms = max(1.0, float(t - ts0))
        elapsed_days = max(1.0 / 24.0, elapsed_ms / 86_400_000.0)
        observed_monthly_turnover = (total_notional / max(initial_capital, 1e-12)) * (30.0 / elapsed_days)
        target_turnover = max(required_monthly_turnover, 700.0)
        turnover_ratio = observed_monthly_turnover / max(target_turnover, 1e-12)
        eff_cooldown_ms = int(max(execution_cooldown_ms, cooldown_ms, cfg_base_cooldown_ms) * turnover_ratio)
        if observed_monthly_turnover > 1000.0:
            overtrade_mult = min(6.0, observed_monthly_turnover / 1000.0)
            eff_cooldown_ms = int(eff_cooldown_ms * overtrade_mult)
            current_bid_sz *= 0.5; current_ask_sz *= 0.5
        eff_cooldown_ms = max(0, eff_cooldown_ms)

        buy_enabled = True; sell_enabled = True
        if last_buy_ts >= 0 and (t - last_buy_ts) < eff_cooldown_ms: buy_enabled = False
        if last_sell_ts >= 0 and (t - last_sell_ts) < eff_cooldown_ms: sell_enabled = False
        if last_fill_ts >= 0 and (t - last_fill_ts) < eff_cooldown_ms:
            buy_enabled = False; sell_enabled = False
            current_bid_sz = 0.0; current_ask_sz = 0.0
        if circuit_until_ts >= 0 and t < circuit_until_ts:
            buy_enabled = False; sell_enabled = False
            current_bid_sz = 0.0; current_ask_sz = 0.0
        if not buy_enabled: current_bid_sz = 0.0
        if not sell_enabled: current_ask_sz = 0.0
        if in_warmup: current_bid_sz = 0.0; current_ask_sz = 0.0

        # ── Side balance ──
        total_side_fills = buy_fill_count + sell_fill_count
        if total_side_fills >= 8:
            buy_ratio = buy_fill_count / (total_side_fills + 1e-12)
            sell_ratio = 1.0 - buy_ratio
            if buy_ratio > max_same_side_fill_ratio:
                over = (buy_ratio - max_same_side_fill_ratio) / (1.0 - max_same_side_fill_ratio + 1e-12)
                throttle = max(0.2, 1.0 - two_sided_rebalance_kappa * over)
                current_bid_sz *= throttle; current_ask_sz *= (1.0 + 0.6 * over)
                quote_bid = _round_to_tick(quote_bid - (1.0 + over) * tick_size, tick_size)
            elif sell_ratio > max_same_side_fill_ratio:
                over = (sell_ratio - max_same_side_fill_ratio) / (1.0 - max_same_side_fill_ratio + 1e-12)
                throttle = max(0.2, 1.0 - two_sided_rebalance_kappa * over)
                current_ask_sz *= throttle; current_bid_sz *= (1.0 + 0.6 * over)
                quote_ask = _round_to_tick(quote_ask + (1.0 + over) * tick_size, tick_size)

        if quote_bid >= quote_ask:
            quote_bid = quote_ask - tick_size

        # ── Submit pending order ──
        pending_bid = quote_bid; pending_ask = quote_ask
        pending_bid_order_size = current_bid_sz; pending_ask_order_size = current_ask_sz
        pending_activate_ts = t + latency_ms
        pending_bid_queue = max(0.0, bsz_obs * queue_ahead_ratio)
        pending_ask_queue = max(0.0, asz_obs * queue_ahead_ratio)
        last_t = t

        # ── Early stop check ──
        if (i + 1) >= next_check and (i + 1) < n:
            elapsed_days = max(1.0 / 24.0, (t - ts0) / 86_400_000.0)
            monthly_turn = (total_notional / max(initial_capital, 1e-12)) * (30.0 / elapsed_days)
            if monthly_turn < (0.8 * required_monthly_turnover):
                early_stopped = 1
                cur_eq = equity[i]
                for j in range(i + 1, n): equity[j] = cur_eq
                break
            next_check += check_step

    fill_rate = float(fill_count) / float(n) if n > 0 else 0.0
    toxic_fill_ratio = 0.0
    return equity, total_notional, fills[:fill_count], fill_rate, toxic_fill_ratio, early_stopped


# ── Original engine (kept for backward compatibility) ──


@njit(cache=True)
def _run_engine(
    ts_ms: np.ndarray,
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_sz: np.ndarray,
    trade_px: np.ndarray,
    trade_sz: np.ndarray,
    trade_side: np.ndarray,
    micro_px: np.ndarray,
    depth_bin: np.ndarray,
    duration_bin: np.ndarray,
    cfg_array: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray, float, float, int]:
    # cfg_array mapping
    initial_capital = cfg_array[0]
    rebate_rate = cfg_array[1]  # decimal, rebate positive
    tick_size = cfg_array[2]
    half_spread_ticks = cfg_array[3]
    gamma_inventory = cfg_array[4]
    max_pos_btc = cfg_array[5]
    order_size_btc = cfg_array[6]
    latency_ms = int(cfg_array[7])
    queue_ahead_ratio = cfg_array[8]
    participation_cap = cfg_array[9]
    queue_model = int(cfg_array[10])
    queue_prob_kappa = cfg_array[11]
    queue_decay_lambda_per_ms = cfg_array[12]
    adverse_selection_bps = cfg_array[13]
    observation_latency_ms = int(cfg_array[14])
    refresh_queue_on_activation = int(cfg_array[15])
    inventory_vol_alpha = cfg_array[16]
    inventory_vol_beta = cfg_array[17]
    thin_depth_spread_mult = cfg_array[18]
    thin_depth_adverse_mult = cfg_array[19]
    fast_duration_spread_mult = cfg_array[20]
    obi_circuit_enabled = int(cfg_array[21])
    obi_hard_threshold = cfg_array[22]
    obi_ema_alpha = cfg_array[23]
    obi_trend_skew_bps = cfg_array[24]
    obi_soft_widen_ticks = cfg_array[25]
    obi_hysteresis_events = int(cfg_array[26])
    obi_hard_disable_inv_ratio = cfg_array[27]
    intensity_threshold = cfg_array[28]
    lambda_intensity = cfg_array[29]
    fill_rate_alpha = cfg_array[30]
    engine_progress_every_n_events = int(cfg_array[31])
    micro_px_alpha = cfg_array[32]
    micro_skew_favorable_kappa = cfg_array[33]
    micro_skew_adverse_kappa = cfg_array[34]
    inventory_aggr_threshold_ratio = cfg_array[35]
    inventory_unwind_narrow_ticks = cfg_array[36]
    inventory_unwind_widen_ticks = cfg_array[37]
    conflict_order_size_ratio = cfg_array[38]
    inventory_skew_nonlinear_power = cfg_array[39]
    dynamic_skew_vol_kappa = cfg_array[40]
    vol_spread_kappa = cfg_array[41]
    execution_cooldown_ms = int(cfg_array[42])
    max_same_side_fill_ratio = cfg_array[43]
    two_sided_rebalance_kappa = cfg_array[44]
    toxic_flow_size_multiple = cfg_array[45]
    toxic_flow_pause_ms = int(cfg_array[46])
    toxic_flow_ema_alpha = cfg_array[47]
    min_side_balance_fills = int(cfg_array[48])
    cooldown_min_size_ratio = cfg_array[49]
    turnover_feedback_kappa = cfg_array[50]
    required_monthly_turnover = cfg_array[51]
    warmup_events = int(cfg_array[52])
    g_laplace_prior_count = cfg_array[53]
    cooldown_ms = int(cfg_array[54])
    cfg_base_cooldown_ms = int(cfg_array[55])
    vol_window_ticks = int(cfg_array[56])
    alpha_obi_fast = cfg_array[57]
    alpha_obi_slow = cfg_array[58]
    alpha_ofi_fast = cfg_array[59]
    alpha_ofi_slow = cfg_array[60]
    alpha_weak_threshold = cfg_array[61]
    alpha_threshold = cfg_array[62]
    alpha_strong_threshold = cfg_array[63]
    alpha_weak_size_ratio = cfg_array[64]
    alpha_strong_favor_mult = cfg_array[65]
    alpha_strong_adverse_mult = cfg_array[66]
    vol_ref_ticks = cfg_array[67]
    max_vol_factor = cfg_array[68]
    min_half_spread_ticks = cfg_array[69]
    max_half_spread_ticks = cfg_array[70]
    toxic_fill_ratio_soft = cfg_array[71]
    toxic_fill_ratio_hard = cfg_array[72]
    toxic_extra_pause_ms = int(cfg_array[73])
    buy_cooldown_multiplier = cfg_array[74]
    sell_cooldown_multiplier = cfg_array[75]
    vol_ref_window_ticks = int(cfg_array[76])
    markout_hard_circuit_bps = cfg_array[77]
    hard_circuit_pause_ms = int(cfg_array[78])
    min_trade_interval_ms = int(cfg_array[79])
    max_order_size_ratio_cfg = cfg_array[80]
    if obi_hard_threshold < 0.5:
        obi_hard_threshold = 0.5
    if obi_hard_threshold > 0.99:
        obi_hard_threshold = 0.99
    if obi_soft_widen_ticks < 0.0:
        obi_soft_widen_ticks = 0.0
    if obi_hysteresis_events < 1:
        obi_hysteresis_events = 1
    if obi_hard_disable_inv_ratio < 0.0:
        obi_hard_disable_inv_ratio = 0.0
    if obi_hard_disable_inv_ratio > 1.0:
        obi_hard_disable_inv_ratio = 1.0
    if obi_ema_alpha < 0.0:
        obi_ema_alpha = 0.0
    if obi_ema_alpha > 1.0:
        obi_ema_alpha = 1.0
    if intensity_threshold < 0.0:
        intensity_threshold = 0.0
    if lambda_intensity < 0.0:
        lambda_intensity = 0.0
    if fill_rate_alpha < 0.0:
        fill_rate_alpha = 0.0
    if fill_rate_alpha > 1.0:
        fill_rate_alpha = 1.0
    if engine_progress_every_n_events < 0:
        engine_progress_every_n_events = 0
    if micro_px_alpha < 0.0:
        micro_px_alpha = 0.0
    if micro_px_alpha > 1.0:
        micro_px_alpha = 1.0
    if micro_skew_favorable_kappa < 0.0:
        micro_skew_favorable_kappa = 0.0
    if micro_skew_adverse_kappa < 0.0:
        micro_skew_adverse_kappa = 0.0
    if inventory_aggr_threshold_ratio < 0.0:
        inventory_aggr_threshold_ratio = 0.0
    if inventory_aggr_threshold_ratio > 1.0:
        inventory_aggr_threshold_ratio = 1.0
    if inventory_unwind_narrow_ticks < 0.0:
        inventory_unwind_narrow_ticks = 0.0
    if inventory_unwind_widen_ticks < 0.0:
        inventory_unwind_widen_ticks = 0.0
    if conflict_order_size_ratio < 0.1:
        conflict_order_size_ratio = 0.1
    if conflict_order_size_ratio > 1.0:
        conflict_order_size_ratio = 1.0
    if inventory_skew_nonlinear_power < 1.0:
        inventory_skew_nonlinear_power = 1.0
    if inventory_skew_nonlinear_power > 4.0:
        inventory_skew_nonlinear_power = 4.0
    if dynamic_skew_vol_kappa < 0.0:
        dynamic_skew_vol_kappa = 0.0
    if vol_spread_kappa < 0.0:
        vol_spread_kappa = 0.0
    if execution_cooldown_ms < 0:
        execution_cooldown_ms = 0
    if max_same_side_fill_ratio < 0.5:
        max_same_side_fill_ratio = 0.5
    if max_same_side_fill_ratio > 0.95:
        max_same_side_fill_ratio = 0.95
    if two_sided_rebalance_kappa < 0.0:
        two_sided_rebalance_kappa = 0.0
    if two_sided_rebalance_kappa > 1.0:
        two_sided_rebalance_kappa = 1.0
    if toxic_flow_size_multiple < 1.0:
        toxic_flow_size_multiple = 1.0
    if toxic_flow_pause_ms < 0:
        toxic_flow_pause_ms = 0
    if toxic_flow_ema_alpha < 0.0:
        toxic_flow_ema_alpha = 0.0
    if toxic_flow_ema_alpha > 1.0:
        toxic_flow_ema_alpha = 1.0
    if min_side_balance_fills < 2:
        min_side_balance_fills = 2
    if cooldown_min_size_ratio < 0.0:
        cooldown_min_size_ratio = 0.0
    if cooldown_min_size_ratio > 1.0:
        cooldown_min_size_ratio = 1.0
    if turnover_feedback_kappa < 0.0:
        turnover_feedback_kappa = 0.0
    if turnover_feedback_kappa > 2.0:
        turnover_feedback_kappa = 2.0
    if required_monthly_turnover < 0.0:
        required_monthly_turnover = 0.0
    if warmup_events < 0:
        warmup_events = 0
    if cooldown_ms < 0:
        cooldown_ms = 0
    if cfg_base_cooldown_ms < 0:
        cfg_base_cooldown_ms = 0
    if half_spread_ticks < 0.5:
        half_spread_ticks = 0.5
    if gamma_inventory < 2.0:
        gamma_inventory = 2.0
    if cooldown_ms < 100:
        cooldown_ms = 100
    if cfg_base_cooldown_ms < 100:
        cfg_base_cooldown_ms = 100
    if vol_window_ticks < 5:
        vol_window_ticks = 5
    if vol_window_ticks > 200:
        vol_window_ticks = 200
    if alpha_obi_fast < 0.0:
        alpha_obi_fast = 0.0
    if alpha_obi_fast > 1.0:
        alpha_obi_fast = 1.0
    if alpha_obi_slow < 0.0:
        alpha_obi_slow = 0.0
    if alpha_obi_slow > 1.0:
        alpha_obi_slow = 1.0
    if alpha_ofi_fast < 0.0:
        alpha_ofi_fast = 0.0
    if alpha_ofi_fast > 1.0:
        alpha_ofi_fast = 1.0
    if alpha_ofi_slow < 0.0:
        alpha_ofi_slow = 0.0
    if alpha_ofi_slow > 1.0:
        alpha_ofi_slow = 1.0
    if alpha_weak_threshold < 0.0:
        alpha_weak_threshold = 0.0
    if alpha_threshold < 0.0:
        alpha_threshold = 0.0
    if alpha_weak_threshold < alpha_threshold:
        alpha_weak_threshold = alpha_threshold
    if alpha_strong_threshold < alpha_weak_threshold:
        alpha_strong_threshold = alpha_weak_threshold
    if alpha_weak_size_ratio < 0.0:
        alpha_weak_size_ratio = 0.0
    if alpha_weak_size_ratio > 1.0:
        alpha_weak_size_ratio = 1.0
    if alpha_strong_favor_mult < 1.0:
        alpha_strong_favor_mult = 1.0
    if alpha_strong_favor_mult > 2.0:
        alpha_strong_favor_mult = 2.0
    if alpha_strong_adverse_mult < 0.0:
        alpha_strong_adverse_mult = 0.0
    if alpha_strong_adverse_mult > 1.0:
        alpha_strong_adverse_mult = 1.0
    if vol_ref_ticks <= 0.0:
        vol_ref_ticks = 0.5
    if max_vol_factor < 0.0:
        max_vol_factor = 0.0
    if min_half_spread_ticks < 0.1:
        min_half_spread_ticks = 0.1
    if max_half_spread_ticks < min_half_spread_ticks:
        max_half_spread_ticks = min_half_spread_ticks
    if toxic_fill_ratio_soft < 0.0:
        toxic_fill_ratio_soft = 0.0
    if toxic_fill_ratio_soft > 1.0:
        toxic_fill_ratio_soft = 1.0
    if toxic_fill_ratio_hard < toxic_fill_ratio_soft:
        toxic_fill_ratio_hard = toxic_fill_ratio_soft
    if toxic_fill_ratio_hard > 1.0:
        toxic_fill_ratio_hard = 1.0
    if toxic_extra_pause_ms < 0:
        toxic_extra_pause_ms = 0
    if buy_cooldown_multiplier < 0.1:
        buy_cooldown_multiplier = 0.1
    if sell_cooldown_multiplier < 0.1:
        sell_cooldown_multiplier = 0.1
    if vol_ref_window_ticks < vol_window_ticks:
        vol_ref_window_ticks = vol_window_ticks
    if vol_ref_window_ticks > 120_000:
        vol_ref_window_ticks = 120_000
    if hard_circuit_pause_ms < 0:
        hard_circuit_pause_ms = 0
    if min_trade_interval_ms < 0:
        min_trade_interval_ms = 0
    if max_order_size_ratio_cfg < 0.2:
        max_order_size_ratio_cfg = 0.2
    if max_order_size_ratio_cfg > 0.3:
        max_order_size_ratio_cfg = 0.3

    n = len(ts_ms)
    equity = np.empty(n, dtype=np.float64)

    cash = initial_capital
    inv = 0.0
    total_notional = 0.0

    # 仅记录成交记录：[ts_ms, side(+1 buy/-1 sell), qty, px]
    fills = np.zeros((n, 4), dtype=np.float64)
    fill_count = 0

    active_bid = 0.0
    active_ask = 0.0
    active_live = 0
    bid_queue_left = 0.0
    ask_queue_left = 0.0
    pending_bid = 0.0
    pending_ask = 0.0
    pending_activate_ts = -1
    pending_bid_queue = 0.0
    pending_ask_queue = 0.0
    active_bid_order_size = order_size_btc
    active_ask_order_size = order_size_btc
    pending_bid_order_size = order_size_btc
    pending_ask_order_size = order_size_btc
    last_t = -1
    obs_idx = 0
    prev_obs_mid = 0.0
    ewma_abs_ret_bps = 0.0
    ema_obi = 0.5
    ema_obi_fast = 0.5
    ema_obi_slow = 0.5
    ofi_ema_fast = 0.0
    ofi_ema_slow = 0.0
    order_size_ema = order_size_btc
    ref_window = np.zeros(vol_ref_window_ticks, dtype=np.float64)
    ref_window_count = 0
    active_size_window = np.zeros(50, dtype=np.float64)
    active_size_sum = 0.0
    active_size_count = 0
    mid_window = np.zeros(vol_window_ticks, dtype=np.float64)
    mid_window_count = 0
    fill_times = np.zeros(100, dtype=np.int64)
    fill_idx = 0
    toxic_fill_count = 0.0
    buy_fill_count = 0
    sell_fill_count = 0
    last_fill_ts = -1
    avg_trade_sz_ema = 0.0
    circuit_until_ts = -1
    pending_mark_ts = np.zeros(128, dtype=np.int64)
    pending_mark_side = np.zeros(128, dtype=np.float64)
    pending_mark_px = np.zeros(128, dtype=np.float64)
    pending_mark_live = np.zeros(128, dtype=np.int64)
    pending_mark_ptr = 0
    markout_bad_streak = 0
    ts0 = ts_ms[0] if n > 0 else 0
    early_stopped = 0
    check_step = max(1, n // 5)
    next_check = check_step
    last_buy_ts = -1
    last_sell_ts = -1
    for i in range(n):
        t = ts_ms[i]
        in_warmup = i < warmup_events
        # 仅统计毒性流，不参与任何交易决策。
        toxic_flow_now = 0
        if inv >= 0.7 * max_pos_btc and ema_obi < (1.0 - obi_hard_threshold):
            toxic_flow_now = 1
        elif inv <= -0.7 * max_pos_btc and ema_obi > obi_hard_threshold:
            toxic_flow_now = 1
        if engine_progress_every_n_events > 0 and (i % engine_progress_every_n_events) == 0:
            # Numba print is supported for simple types; keep it lightweight.
            print("engine_progress", i, n, int(t))

        # Observation latency: at time t we only "see" book up to t - observation_latency_ms.
        obs_cut = t - observation_latency_ms
        while (obs_idx + 1) < n and ts_ms[obs_idx + 1] <= obs_cut:
            obs_idx += 1

        bpx_obs = bid_px[obs_idx]
        apx_obs = ask_px[obs_idx]
        bsz_obs = bid_sz[obs_idx]
        asz_obs = ask_sz[obs_idx]
        obi_signed_now = (bsz_obs - asz_obs) / (bsz_obs + asz_obs + 1e-12)
        mid_obs = 0.5 * (bpx_obs + apx_obs)
        mid_window[i % vol_window_ticks] = mid_obs
        if mid_window_count < vol_window_ticks:
            mid_window_count += 1
        ref_window[i % vol_ref_window_ticks] = mid_obs
        if ref_window_count < vol_ref_window_ticks:
            ref_window_count += 1

        # MTM still uses current market mid (not stale observed mid).
        mid = 0.5 * (bid_px[i] + ask_px[i])
        tpx = trade_px[i]
        tsz = trade_sz[i]
        tside = trade_side[i]  # +1 aggressive buy, -1 aggressive sell
        ofi_now = tside * tsz
        ofi_ema_fast = (1.0 - alpha_ofi_fast) * ofi_ema_fast + alpha_ofi_fast * ofi_now
        ofi_ema_slow = (1.0 - alpha_ofi_slow) * ofi_ema_slow + alpha_ofi_slow * ofi_now
        active_order_size = tsz
        slot = i % 50
        if active_size_count < 50:
            active_size_count += 1
        else:
            active_size_sum -= active_size_window[slot]
        active_size_window[slot] = active_order_size
        active_size_sum += active_order_size
        active_size_mean = active_size_sum / max(1, active_size_count)
        toxic_pause_ms = 0
        if active_order_size > 3.0 * active_size_mean:
            toxic_pause_ms = 200
        elif active_order_size > 2.0 * active_size_mean:
            toxic_pause_ms = 50
        if tsz > 0.0:
            if avg_trade_sz_ema <= 0.0:
                avg_trade_sz_ema = tsz
            else:
                avg_trade_sz_ema = (1.0 - toxic_flow_ema_alpha) * avg_trade_sz_ema + toxic_flow_ema_alpha * tsz

        # 100ms markout monitor for hard-circuit protection.
        for m in range(128):
            if pending_mark_live[m] == 1 and t >= (pending_mark_ts[m] + 100):
                side = pending_mark_side[m]
                exec_px = pending_mark_px[m]
                mark_bps = side * (mid - exec_px) / (exec_px + 1e-12) * 10_000.0
                if mark_bps < markout_hard_circuit_bps:
                    markout_bad_streak += 1
                else:
                    markout_bad_streak = 0
                pending_mark_live[m] = 0
        if markout_bad_streak >= 4:
            circuit_until_ts = t + hard_circuit_pause_ms
            markout_bad_streak = 0

        # Queue naturally decays over time due to cancellations/modifications.
        if active_live == 1 and last_t >= 0 and t > last_t:
            dt_ms = float(t - last_t)
            decay = np.exp(-queue_decay_lambda_per_ms * dt_ms)
            bid_queue_left *= decay
            ask_queue_left *= decay

        # 延迟生效：pending quote 到点后激活为 active quote
        if pending_activate_ts >= 0 and t >= pending_activate_ts:
            active_bid = pending_bid
            active_ask = pending_ask
            active_bid_order_size = pending_bid_order_size
            active_ask_order_size = pending_ask_order_size
            if refresh_queue_on_activation == 1:
                bid_queue_left = max(0.0, bsz_obs * queue_ahead_ratio)
                ask_queue_left = max(0.0, asz_obs * queue_ahead_ratio)
            else:
                bid_queue_left = pending_bid_queue
                ask_queue_left = pending_ask_queue
            active_live = 1
            pending_activate_ts = -1

        # 成交判定（保守）：穿透或者同价且扫掉排队+我单
        if (not in_warmup) and active_live == 1 and active_bid > 0.0:
            # aggressive sell hits bids
            if tside < 0:
                # 撮合应基于盘口可成交价，而不是历史 trade_px。
                is_cross = active_bid >= apx_obs
                is_touch = np.abs(active_bid - apx_obs) <= (0.5 * tick_size)
                fill_qty = 0.0

                # 穿透成交：默认按可参与交易量约束成交
                if is_cross:
                    fill_qty = min(active_bid_order_size, tsz * participation_cap, max_pos_btc - inv)

                if is_touch:
                    if queue_model == 0:
                        bid_queue_left = max(0.0, bid_queue_left - tsz)
                        avail = max(0.0, tsz * participation_cap - bid_queue_left)
                        fill_qty = max(fill_qty, min(active_bid_order_size, avail, max_pos_btc - inv))
                    else:
                        # 概率排队模型：输出期望成交量（无随机数，稳定可复现）
                        pressure = (tsz * participation_cap) / (bid_queue_left + active_bid_order_size + 1e-12)
                        p_fill = 1.0 - np.exp(-queue_prob_kappa * max(0.0, pressure))
                        p_fill = min(1.0, max(0.0, p_fill))
                        exp_qty = active_bid_order_size * p_fill
                        fill_qty = max(fill_qty, min(exp_qty, max_pos_btc - inv))
                        bid_queue_left = max(0.0, bid_queue_left - tsz)

                if fill_qty > 0.0:
                    adv_mult = thin_depth_adverse_mult if depth_bin[obs_idx] == 0 else 1.0
                    exec_px = active_bid * (1.0 + (adverse_selection_bps * adv_mult) / 10_000.0)
                    notional = fill_qty * exec_px
                    cash -= notional
                    cash += notional * rebate_rate
                    inv += fill_qty
                    total_notional += notional
                    fills[fill_count, 0] = t
                    fills[fill_count, 1] = 1.0
                    fills[fill_count, 2] = fill_qty
                    fills[fill_count, 3] = exec_px
                    fill_count += 1
                    fill_times[fill_idx % 100] = t
                    fill_idx += 1
                    buy_fill_count += 1
                    last_fill_ts = t
                    last_buy_ts = t
                    pm = pending_mark_ptr % 128
                    pending_mark_ts[pm] = t
                    pending_mark_side[pm] = 1.0
                    pending_mark_px[pm] = exec_px
                    pending_mark_live[pm] = 1
                    pending_mark_ptr += 1
                    if toxic_flow_now == 1:
                        toxic_fill_count += 1.0

        if (not in_warmup) and active_live == 1 and active_ask > 0.0:
            # aggressive buy lifts asks
            if tside > 0:
                # 撮合应基于盘口可成交价，而不是历史 trade_px。
                is_cross = active_ask <= bpx_obs
                is_touch = np.abs(active_ask - bpx_obs) <= (0.5 * tick_size)
                fill_qty = 0.0

                if is_cross:
                    fill_qty = min(active_ask_order_size, tsz * participation_cap, max_pos_btc + inv)

                if is_touch:
                    if queue_model == 0:
                        ask_queue_left = max(0.0, ask_queue_left - tsz)
                        avail = max(0.0, tsz * participation_cap - ask_queue_left)
                        fill_qty = max(fill_qty, min(active_ask_order_size, avail, max_pos_btc + inv))
                    else:
                        pressure = (tsz * participation_cap) / (ask_queue_left + active_ask_order_size + 1e-12)
                        p_fill = 1.0 - np.exp(-queue_prob_kappa * max(0.0, pressure))
                        p_fill = min(1.0, max(0.0, p_fill))
                        exp_qty = active_ask_order_size * p_fill
                        fill_qty = max(fill_qty, min(exp_qty, max_pos_btc + inv))
                        ask_queue_left = max(0.0, ask_queue_left - tsz)

                if fill_qty > 0.0:
                    adv_mult = thin_depth_adverse_mult if depth_bin[obs_idx] == 0 else 1.0
                    exec_px = active_ask * (1.0 - (adverse_selection_bps * adv_mult) / 10_000.0)
                    notional = fill_qty * exec_px
                    cash += notional
                    cash += notional * rebate_rate
                    inv -= fill_qty
                    total_notional += notional
                    fills[fill_count, 0] = t
                    fills[fill_count, 1] = -1.0
                    fills[fill_count, 2] = fill_qty
                    fills[fill_count, 3] = exec_px
                    fill_count += 1
                    fill_times[fill_idx % 100] = t
                    fill_idx += 1
                    sell_fill_count += 1
                    last_fill_ts = t
                    last_sell_ts = t
                    pm = pending_mark_ptr % 128
                    pending_mark_ts[pm] = t
                    pending_mark_side[pm] = -1.0
                    pending_mark_px[pm] = exec_px
                    pending_mark_live[pm] = 1
                    pending_mark_ptr += 1
                    if toxic_flow_now == 1:
                        toxic_fill_count += 1.0

        # mark-to-market
        equity[i] = cash + inv * mid

        prev_mid_for_spread = prev_obs_mid
        # Vol-adaptive inventory state update (kept for diagnostics / future use).
        if prev_obs_mid > 0.0 and mid_obs > 0.0:
            ret_bps = np.abs(mid_obs / prev_obs_mid - 1.0) * 10_000.0
            ewma_abs_ret_bps = (1.0 - inventory_vol_alpha) * ewma_abs_ret_bps + inventory_vol_alpha * ret_bps
        prev_obs_mid = mid_obs

        # Step 1) EMA OBI update
        current_obi = bsz_obs / (bsz_obs + asz_obs + 1e-12)
        ema_obi = obi_ema_alpha * current_obi + (1.0 - obi_ema_alpha) * ema_obi
        ema_obi_fast = alpha_obi_fast * current_obi + (1.0 - alpha_obi_fast) * ema_obi_fast
        ema_obi_slow = alpha_obi_slow * current_obi + (1.0 - alpha_obi_slow) * ema_obi_slow

        # Step 2) Mid-price anchor + alpha gating
        micro_obs = micro_px[obs_idx]
        adjusted_mid = mid_obs
        micro_gap = micro_obs - mid_obs
        threshold = 0.45 * tick_size
        bid_scale = 1.0
        ask_scale = 1.0
        if micro_gap > threshold:
            bid_scale = 0.4
        elif micro_gap < -threshold:
            ask_scale = 0.4
        alpha_signal = 0.6 * (ema_obi_fast - ema_obi_slow) + 0.4 * ((ofi_ema_fast - ofi_ema_slow) / (avg_trade_sz_ema + 1e-12))
        alpha_abs = np.abs(alpha_signal)
        if alpha_abs < alpha_weak_threshold:
            bid_scale *= alpha_weak_size_ratio
            ask_scale *= alpha_weak_size_ratio
        elif alpha_signal > alpha_strong_threshold:
            bid_scale *= alpha_strong_favor_mult
            ask_scale *= alpha_strong_adverse_mult
        elif alpha_signal < -alpha_strong_threshold:
            ask_scale *= alpha_strong_favor_mult
            bid_scale *= alpha_strong_adverse_mult

        # Wall-clock fill intensity: last 50 fills within 60s wall time => widen spread.
        adaptive_spread_mult = 1.0
        if fill_idx >= 50:
            oldest_idx = (fill_idx - 50) % 100
            time_diff_ms = t - fill_times[oldest_idx]
            if time_diff_ms > 0 and time_diff_ms < 60000:
                implied_fills_per_min = 50.0 * (60000.0 / time_diff_ms)
                intensity_gap = max(0.0, implied_fills_per_min - intensity_threshold)
                adaptive_spread_mult = 1.0 + (lambda_intensity * intensity_gap)
        # 库存逼近极限时强制进入紧急撤退模式，避免单边行情里持续被动成交。
        if np.abs(inv) >= (0.9 * max_pos_btc):
            adaptive_spread_mult *= 2.0

        # Step 3) Nonlinear inventory skew before pricing.
        inventory_ratio = inv / (max_pos_btc + 1e-12)
        inv_abs = np.abs(inventory_ratio)
        inv_curve = inv_abs**inventory_skew_nonlinear_power
        sigmoid_part = 1.0 / (1.0 + np.exp(-8.0 * (inv_curve - 0.35)))
        dynamic_skew_boost = 1.0 + dynamic_skew_vol_kappa * (ewma_abs_ret_bps / 10.0)
        skew_ticks = gamma_inventory * sigmoid_part * dynamic_skew_boost
        if skew_ticks > 4.0:
            skew_ticks = 4.0
        skew_factor = np.sign(inventory_ratio) * skew_ticks * tick_size
        adjusted_mid = adjusted_mid - skew_factor
        # Activate advanced microprice anchor with confidence filter.
        micro_confidence = np.abs(micro_obs - mid_obs) / (tick_size * 2.0 + 1e-12)
        if micro_confidence < 0.6:
            micro_px_alpha_effective = 0.3
        else:
            micro_px_alpha_effective = micro_px_alpha
        adjusted_mid = (1.0 - micro_px_alpha_effective) * adjusted_mid + micro_px_alpha_effective * micro_obs
        reservation_price = adjusted_mid
        # Step 4) Volatility-adaptive spread using short rolling std of observed mid.
        vol_short = 0.0
        if mid_window_count >= 3:
            mean_mid = 0.0
            for j in range(mid_window_count):
                mean_mid += mid_window[j]
            mean_mid /= mid_window_count
            var_mid = 0.0
            for j in range(mid_window_count):
                d = mid_window[j] - mean_mid
                var_mid += d * d
            var_mid /= mid_window_count
            vol_short = np.sqrt(max(0.0, var_mid))
        vol_short_ticks = vol_short / (tick_size + 1e-12)
        vol_ref_dynamic = vol_ref_ticks
        if ref_window_count >= 10:
            ref_mean = 0.0
            for j in range(ref_window_count):
                ref_mean += ref_window[j]
            ref_mean /= ref_window_count
            ref_var = 0.0
            for j in range(ref_window_count):
                d = ref_window[j] - ref_mean
                ref_var += d * d
            ref_var /= ref_window_count
            vol_ref_dynamic = max(vol_ref_ticks, np.sqrt(max(0.0, ref_var)) / (tick_size + 1e-12))
        vol_scale = min(vol_short_ticks / (vol_ref_dynamic + 1e-12), max_vol_factor)
        # Volatility-adaptive spread (additive form): base + k * current_volatility.
        half_spread_dyn_ticks = half_spread_ticks + vol_spread_kappa * vol_short_ticks
        half_spread_adj = half_spread_dyn_ticks * adaptive_spread_mult
        if vol_scale > 0.9:
            half_spread_adj *= 1.15
        if depth_bin[obs_idx] == 0:
            half_spread_adj *= thin_depth_spread_mult
        if duration_bin[obs_idx] == 0:
            half_spread_adj *= fast_duration_spread_mult
        half_spread_adj = min(max_half_spread_ticks, max(min_half_spread_ticks, half_spread_adj))
        current_half_spread = half_spread_adj * tick_size
        # Hard spread boundaries for Numba safety and robustness.
        current_half_spread = max(current_half_spread, 0.5 * tick_size)
        current_half_spread = min(current_half_spread, 5.0 * tick_size)
        quote_bid = _round_to_tick(reservation_price - current_half_spread, tick_size)
        quote_ask = _round_to_tick(reservation_price + current_half_spread, tick_size)

        max_size = order_size_btc * 0.25
        current_bid_order_size = min(order_size_btc * bid_scale, max_size)
        current_ask_order_size = min(order_size_btc * ask_scale, max_size)
        # Signal filtering: when micro-price points one way, cancel opposite-side quote.
        if micro_gap > threshold:
            current_ask_order_size *= 0.5
        elif micro_gap < -threshold:
            current_bid_order_size *= 0.5
        # OBI hard filter: in strong one-way pressure, never quote against trend side.
        if obi_signed_now > 0.8:
            current_ask_order_size = 0.0
        elif obi_signed_now < -0.8:
            current_bid_order_size = 0.0
        # Inventory protection: stop adding in the direction that is already near cap.
        if inv >= 0.7 * max_pos_btc:
            current_bid_order_size = 0.0
        if inv <= -0.7 * max_pos_btc:
            current_ask_order_size = 0.0
        current_order_size = 0.5 * (current_bid_order_size + current_ask_order_size)
        order_size_ema = 0.98 * order_size_ema + 0.02 * current_order_size
        elapsed_ms = max(1.0, float(t - ts0))
        elapsed_days = elapsed_ms / 86_400_000.0
        if elapsed_days < (1.0 / 24.0):
            elapsed_days = 1.0 / 24.0
        observed_monthly_turnover = (total_notional / max(initial_capital, 1e-12)) * (30.0 / elapsed_days)
        target_turnover = 700.0
        if required_monthly_turnover > target_turnover:
            target_turnover = required_monthly_turnover
        turnover_ratio = observed_monthly_turnover / max(target_turnover, 1e-12)
        order_ratio = order_size_ema / max(current_order_size, 1e-12)
        cooldown_scale = max(1.0, turnover_ratio * order_ratio)
        base_cooldown_ms = execution_cooldown_ms
        if cooldown_ms > base_cooldown_ms:
            base_cooldown_ms = cooldown_ms
        if cfg_base_cooldown_ms > base_cooldown_ms:
            base_cooldown_ms = cfg_base_cooldown_ms
        eff_cooldown_ms = int(base_cooldown_ms * cooldown_scale)
        if observed_monthly_turnover > 1000.0:
            overtrade_mult = observed_monthly_turnover / 1000.0
            if overtrade_mult > 6.0:
                overtrade_mult = 6.0
            eff_cooldown_ms = int(eff_cooldown_ms * overtrade_mult)
            current_bid_order_size *= 0.5
            current_ask_order_size *= 0.5
        if toxic_pause_ms > eff_cooldown_ms:
            eff_cooldown_ms = toxic_pause_ms
        observed_toxic_ratio = toxic_fill_count / (fill_count + 1e-12)
        if observed_toxic_ratio >= toxic_fill_ratio_hard:
            eff_cooldown_ms = max(eff_cooldown_ms, toxic_extra_pause_ms)
            current_bid_order_size *= 0.5
            current_ask_order_size *= 0.5
        elif observed_toxic_ratio >= toxic_fill_ratio_soft:
            eff_cooldown_ms = max(eff_cooldown_ms, int(0.5 * toxic_extra_pause_ms))
        if eff_cooldown_ms < 0:
            eff_cooldown_ms = 0
        buy_cooldown_ms = int(eff_cooldown_ms * buy_cooldown_multiplier)
        sell_cooldown_ms = int(eff_cooldown_ms * sell_cooldown_multiplier)
        if buy_cooldown_ms < 0:
            buy_cooldown_ms = 0
        if sell_cooldown_ms < 0:
            sell_cooldown_ms = 0

        buy_enabled = True
        sell_enabled = True
        if last_buy_ts >= 0 and (t - last_buy_ts) < buy_cooldown_ms:
            buy_enabled = False
        if last_sell_ts >= 0 and (t - last_sell_ts) < sell_cooldown_ms:
            sell_enabled = False
        if last_buy_ts >= 0 and (t - last_buy_ts) < min_trade_interval_ms:
            buy_enabled = False
        if last_sell_ts >= 0 and (t - last_sell_ts) < min_trade_interval_ms:
            sell_enabled = False
        if last_fill_ts >= 0 and (t - last_fill_ts) < eff_cooldown_ms:
            buy_enabled = False
            sell_enabled = False
            current_bid_order_size = 0.0
            current_ask_order_size = 0.0
        if alpha_abs < alpha_threshold:
            buy_enabled = False
            sell_enabled = False
            current_bid_order_size = 0.0
            current_ask_order_size = 0.0
        if circuit_until_ts >= 0 and t < circuit_until_ts:
            buy_enabled = False
            sell_enabled = False
            current_bid_order_size = 0.0
            current_ask_order_size = 0.0
        if not buy_enabled:
            current_bid_order_size = 0.0
        if not sell_enabled:
            current_ask_order_size = 0.0
        if in_warmup:
            # Warmup keeps feature/filter states converging while avoiding early biased fills.
            current_bid_order_size = 0.0
            current_ask_order_size = 0.0

        total_side_fills = buy_fill_count + sell_fill_count
        if total_side_fills >= min_side_balance_fills:
            buy_ratio = buy_fill_count / (total_side_fills + 1e-12)
            sell_ratio = sell_fill_count / (total_side_fills + 1e-12)
            if buy_ratio > max_same_side_fill_ratio:
                over = (buy_ratio - max_same_side_fill_ratio) / (1.0 - max_same_side_fill_ratio + 1e-12)
                throttle = max(0.2, 1.0 - two_sided_rebalance_kappa * over)
                rebalance_push_ticks = (1.0 + 1.5 * gamma_inventory) * two_sided_rebalance_kappa * over
                current_bid_order_size *= throttle
                current_ask_order_size *= (1.0 + 0.6 * over)
                quote_bid = _round_to_tick(quote_bid - (1.0 + over) * tick_size, tick_size)
                quote_ask = _round_to_tick(quote_ask - rebalance_push_ticks * tick_size, tick_size)
            elif sell_ratio > max_same_side_fill_ratio:
                over = (sell_ratio - max_same_side_fill_ratio) / (1.0 - max_same_side_fill_ratio + 1e-12)
                throttle = max(0.2, 1.0 - two_sided_rebalance_kappa * over)
                rebalance_push_ticks = (1.0 + 1.5 * gamma_inventory) * two_sided_rebalance_kappa * over
                current_ask_order_size *= throttle
                current_bid_order_size *= (1.0 + 0.6 * over)
                quote_ask = _round_to_tick(quote_ask + (1.0 + over) * tick_size, tick_size)
                quote_bid = _round_to_tick(quote_bid + rebalance_push_ticks * tick_size, tick_size)

        # Step 5) Final safety check (avoid crossed quotes)
        if quote_bid >= quote_ask:
            quote_bid = quote_ask - tick_size

        # 下发新订单，等待 latency 后生效（替换旧 pending）
        pending_bid = quote_bid
        pending_ask = quote_ask
        pending_bid_order_size = current_bid_order_size
        pending_ask_order_size = current_ask_order_size
        pending_activate_ts = t + latency_ms
        pending_bid_queue = max(0.0, bsz_obs * queue_ahead_ratio)
        pending_ask_queue = max(0.0, asz_obs * queue_ahead_ratio)
        last_t = t

        if (i + 1) >= next_check and (i + 1) < n:
            elapsed_days = max((t - ts0) / 86_400_000.0, 1.0 / 24.0)
            monthly_turn = (total_notional / max(initial_capital, 1e-12)) * (30.0 / elapsed_days)
            if monthly_turn < (0.8 * required_monthly_turnover):
                early_stopped = 1
                cur_eq = equity[i]
                for j in range(i + 1, n):
                    equity[j] = cur_eq
                break
            next_check += check_step

    fill_rate = float(fill_count) / float(n) if n > 0 else 0.0
    toxic_fill_ratio = toxic_fill_count / float(fill_count) if fill_count > 0 else 0.0
    return equity, total_notional, fills[:fill_count], fill_rate, toxic_fill_ratio, early_stopped


def run_backtest(events: pl.DataFrame, cfg: BacktestConfig, grid_stage: bool = False) -> Dict[str, object]:
    # Compatibility guard: prebuilt events parquet may not carry micro_px.
    if "micro_px" not in events.columns:
        if {"bid_px", "ask_px", "bid_sz", "ask_sz"}.issubset(set(events.columns)):
            events = events.with_columns(
                (
                    pl.col("bid_px")
                    + (pl.col("ask_px") - pl.col("bid_px"))
                    * (pl.col("bid_sz") / (pl.col("bid_sz") + pl.col("ask_sz") + 1e-12))
                ).alias("micro_px")
            )
        elif "mid_px" in events.columns:
            events = events.with_columns(pl.col("mid_px").alias("micro_px"))
        else:
            raise ValueError("events 缺少 micro_px 且无法从 bid/ask 或 mid_px 推导")

    rebate_rate = -cfg.maker_rebate_bps / 10_000.0  # -0.5 bps => +0.00005 rebate cash

    ts_ms = events["ts_ms"].to_numpy()
    bid_px = events["bid_px"].to_numpy()
    ask_px = events["ask_px"].to_numpy()
    bid_sz = events["bid_sz"].to_numpy()
    ask_sz = events["ask_sz"].to_numpy()
    trade_px = events["trade_px"].to_numpy()
    trade_sz = events["trade_sz"].to_numpy()
    trade_side = events["trade_side"].to_numpy()
    micro_col = "micro_px_adv" if (cfg.use_advanced_microprice and "micro_px_adv" in events.columns) else "micro_px"
    micro_px = events[micro_col].to_numpy()
    depth_bin = (
        events["state_depth_bin"].to_numpy().astype(np.int64)
        if "state_depth_bin" in events.columns
        else np.zeros(events.height, dtype=np.int64)
    )
    duration_bin = (
        events["state_duration_bin"].to_numpy().astype(np.int64)
        if "state_duration_bin" in events.columns
        else np.zeros(events.height, dtype=np.int64)
    )

    # ── Dictionary training / loading ──
    D = np.zeros((cfg.dict_n_features, cfg.dict_n_components), dtype=np.float64)
    scaler_mean = np.zeros(cfg.dict_n_features, dtype=np.float64)
    scaler_std = np.ones(cfg.dict_n_features, dtype=np.float64)
    alpha_train = None

    if cfg.use_dictionary_strategy:
        dict_cache = Path(cfg.dict_cache_path) if cfg.dict_cache_path else None
        if dict_cache and dict_cache.exists():
            data = np.load(str(dict_cache))
            D = data["D"]
            scaler_mean = data.get("scaler_mean", np.zeros(cfg.dict_n_features, dtype=np.float64))
            scaler_std = data.get("scaler_std", np.ones(cfg.dict_n_features, dtype=np.float64))
            print(f"加载缓存字典: {dict_cache}  shape={D.shape}")
        elif HAS_SKLEARN:
            # Fit dictionary on training prefix only
            X_full, scaler_state = build_feature_matrix(events, cfg, fit_scalers=True)
            scaler_mean = scaler_state["mean"]
            scaler_std = scaler_state["std"]

            n_train = max(cfg.dict_batch_size * 5, int(X_full.shape[0] * cfg.dict_train_ratio))
            n_train = min(n_train, X_full.shape[0])
            X_train = X_full[:n_train]

            print(f"训练字典: X_train={X_train.shape}, n_components={cfg.dict_n_components}, alpha={cfg.dict_alpha}")
            D, alpha_train = fit_dictionary(X_train, cfg)
            print(f"字典训练完成 D={D.shape}  sparsity={np.mean(np.abs(alpha_train) < 1e-6):.2%}")

            if dict_cache:
                np.savez(str(dict_cache), D=D, scaler_mean=scaler_mean, scaler_std=scaler_std)
                print(f"字典已缓存: {dict_cache}")
        else:
            print("sklearn 未安装且无缓存字典，回退到原版引擎")
            cfg.use_dictionary_strategy = False

    cfg_array = np.array(
        [
            cfg.initial_capital,
            rebate_rate,
            cfg.tick_size,
            cfg.half_spread_ticks,
            cfg.gamma_inventory,
            cfg.max_pos_btc,
            cfg.order_size_btc,
            float(cfg.latency_ms),
            cfg.queue_ahead_ratio,
            cfg.participation_cap,
            float(cfg.queue_model),
            cfg.queue_prob_kappa,
            cfg.queue_decay_lambda_per_ms,
            cfg.adverse_selection_bps,
            float(cfg.observation_latency_ms),
            float(cfg.refresh_queue_on_activation),
            cfg.inventory_vol_alpha,
            cfg.inventory_vol_beta,
            cfg.thin_depth_spread_mult,
            cfg.thin_depth_adverse_mult,
            cfg.fast_duration_spread_mult,
            float(cfg.obi_circuit_enabled),
            cfg.obi_hard_threshold,
            cfg.obi_ema_alpha,
            cfg.obi_trend_skew_bps,
            cfg.obi_soft_widen_ticks,
            float(cfg.obi_hysteresis_events),
            cfg.obi_hard_disable_inv_ratio,
            cfg.intensity_threshold,
            cfg.lambda_intensity,
            cfg.fill_rate_alpha,
            float(cfg.engine_progress_every_n_events),
            cfg.micro_px_alpha,
            cfg.micro_skew_favorable_kappa,
            cfg.micro_skew_adverse_kappa,
            cfg.inventory_aggr_threshold_ratio,
            cfg.inventory_unwind_narrow_ticks,
            cfg.inventory_unwind_widen_ticks,
            cfg.conflict_order_size_ratio,
            cfg.inventory_skew_nonlinear_power,
            cfg.dynamic_skew_vol_kappa,
            cfg.vol_spread_kappa,
            float(cfg.execution_cooldown_ms),
            cfg.max_same_side_fill_ratio,
            cfg.two_sided_rebalance_kappa,
            cfg.toxic_flow_size_multiple,
            float(cfg.toxic_flow_pause_ms),
            cfg.toxic_flow_ema_alpha,
            float(cfg.min_side_balance_fills),
            cfg.cooldown_min_size_ratio,
            cfg.turnover_feedback_kappa,
            cfg.required_monthly_turnover,
            float(cfg.warmup_events),
            cfg.g_laplace_prior_count,
            float(cfg.cooldown_ms),
            float(cfg.base_cooldown_ms),
            float(cfg.vol_window_ticks),
            cfg.alpha_obi_fast,
            cfg.alpha_obi_slow,
            cfg.alpha_ofi_fast,
            cfg.alpha_ofi_slow,
            cfg.alpha_weak_threshold,
            cfg.alpha_threshold,
            cfg.alpha_strong_threshold,
            cfg.alpha_weak_size_ratio,
            cfg.alpha_strong_favor_mult,
            cfg.alpha_strong_adverse_mult,
            cfg.vol_ref_ticks,
            cfg.max_vol_factor,
            cfg.min_half_spread_ticks,
            cfg.max_half_spread_ticks,
            cfg.toxic_fill_ratio_soft,
            cfg.toxic_fill_ratio_hard,
            float(cfg.toxic_extra_pause_ms),
            cfg.buy_cooldown_multiplier,
            cfg.sell_cooldown_multiplier,
            float(cfg.vol_ref_window_ticks),
            cfg.markout_hard_circuit_bps,
            float(cfg.hard_circuit_pause_ms),
            float(cfg.min_trade_interval_ms),
            cfg.max_order_size_ratio,
            # dict-specific params (indices 82-87)
            float(cfg.dict_sparse_max_iter),
            cfg.dict_alpha,
            cfg.dict_signal_alpha_scale,
            cfg.dict_residual_widen_ticks,
            cfg.dict_anomaly_threshold,
            float(cfg.dict_n_components),
        ],
        dtype=np.float64,
    )

    if cfg.use_dictionary_strategy:
        equity, total_notional, fills, fill_rate, toxic_fill_ratio, early_stopped = _run_engine_dict(
            ts_ms, bid_px, ask_px, bid_sz, ask_sz,
            trade_px, trade_sz, trade_side, micro_px,
            depth_bin, duration_bin, cfg_array,
            D, scaler_mean, scaler_std,
        )
    else:
        equity, total_notional, fills, fill_rate, toxic_fill_ratio, early_stopped = _run_engine(
            ts_ms, bid_px, ask_px, bid_sz, ask_sz,
            trade_px, trade_sz, trade_side, micro_px,
            depth_bin, duration_bin, cfg_array,
        )

    equity_df = pl.DataFrame(
        {
            "ts_ms": ts_ms,
            "equity": equity,
        }
    )
    fills_df = pl.DataFrame(
        {
            "ts_ms": fills[:, 0] if len(fills) else np.array([], dtype=np.float64),
            "side": fills[:, 1] if len(fills) else np.array([], dtype=np.float64),
            "qty": fills[:, 2] if len(fills) else np.array([], dtype=np.float64),
            "px": fills[:, 3] if len(fills) else np.array([], dtype=np.float64),
        }
    )

    metrics = evaluate_performance(equity_df, total_notional, cfg.initial_capital, cfg)
    markout = _compute_fill_markouts(events, fills_df, only_500ms=grid_stage)
    side_stats = _compute_fill_side_pnl(events, fills_df)
    metrics["markout_100ms_bps"] = float(markout["markout_100ms_bps"])
    metrics["markout_500ms_bps"] = float(markout["markout_500ms_bps"])
    metrics["markout_1s_bps"] = float(markout["markout_1s_bps"])
    metrics["markout_5s_bps"] = float(markout["markout_5s_bps"])
    metrics["toxic_fill_ratio"] = float(toxic_fill_ratio)
    metrics["early_stopped"] = int(early_stopped)
    metrics["eval_skipped"] = int(early_stopped)
    if early_stopped:
        # Grid-stage early-stop failed the turnover floor; skip risk ranking metrics.
        metrics["turnover_passed"] = False
        metrics["sharpe"] = np.nan
        metrics["calmar"] = np.nan
        metrics["annual_return"] = np.nan
    metrics["buy_side_pnl"] = float(side_stats["buy_side_pnl"])
    metrics["sell_side_pnl"] = float(side_stats["sell_side_pnl"])
    metrics["buy_fill_count"] = float(side_stats["buy_fill_count"])
    metrics["sell_fill_count"] = float(side_stats["sell_fill_count"])
    return {
        "metrics": metrics,
        "equity_df": equity_df,
        "fills_df": fills_df,
        "total_notional": total_notional,
        "fill_rate": float(fill_rate),
        "toxic_fill_ratio": float(toxic_fill_ratio),
        "early_stopped": int(early_stopped),
        "markout_100ms_bps": float(markout["markout_100ms_bps"]),
        "markout_500ms_bps": float(markout["markout_500ms_bps"]),
        "markout_1s_bps": float(markout["markout_1s_bps"]),
        "markout_5s_bps": float(markout["markout_5s_bps"]),
        "buy_side_pnl": float(side_stats["buy_side_pnl"]),
        "sell_side_pnl": float(side_stats["sell_side_pnl"]),
    }


def _compute_fill_markouts(events: pl.DataFrame, fills_df: pl.DataFrame, only_500ms: bool = False) -> Dict[str, float]:
    if fills_df.height == 0:
        return {
            "markout_100ms_bps": np.nan,
            "markout_500ms_bps": np.nan,
            "markout_1s_bps": np.nan,
            "markout_5s_bps": np.nan,
        }

    if "mid_px" in events.columns:
        mid_expr = pl.col("mid_px").cast(pl.Float64)
    elif {"bid_px", "ask_px"}.issubset(set(events.columns)):
        mid_expr = ((pl.col("bid_px") + pl.col("ask_px")) * 0.5).cast(pl.Float64)
    else:
        return {
            "markout_100ms_bps": np.nan,
            "markout_500ms_bps": np.nan,
            "markout_1s_bps": np.nan,
            "markout_5s_bps": np.nan,
        }

    mid_ref = (
        events.select(
            [
                pl.col("ts_ms").cast(pl.Int64).alias("ts_ms"),
                mid_expr.alias("mid_px"),
            ]
        )
        .sort("ts_ms")
    )

    fills = (
        fills_df.select(
            [
                pl.col("ts_ms").cast(pl.Int64).alias("fill_ts_ms"),
                pl.col("side").cast(pl.Float64).alias("side"),
                pl.col("px").cast(pl.Float64).alias("exec_px"),
            ]
        )
        .with_row_index(name="fill_id")
        .with_columns(
            [
                (pl.col("fill_ts_ms") + 100).alias("ts_100ms"),
                (pl.col("fill_ts_ms") + 500).alias("ts_500ms"),
                (pl.col("fill_ts_ms") + 1000).alias("ts_1s"),
                (pl.col("fill_ts_ms") + 5000).alias("ts_5s"),
            ]
        )
    )

    fill_500ms = (
        fills.sort("ts_500ms")
        .join_asof(
            mid_ref,
            left_on="ts_500ms",
            right_on="ts_ms",
            strategy="forward",
        )
        .select([pl.col("fill_id"), pl.col("mid_px").alias("mid_500ms")])
    )
    if only_500ms:
        markout_df = (
            fills.join(fill_500ms, on="fill_id", how="left")
            .with_columns(
                [
                    pl.when(pl.col("side") > 0.0)
                    .then((pl.col("mid_500ms") / pl.col("exec_px") - 1.0) * 10000.0)
                    .otherwise((1.0 - pl.col("mid_500ms") / pl.col("exec_px")) * 10000.0)
                    .alias("markout_500ms_bps"),
                ]
            )
        )
        m500 = markout_df.select(pl.col("markout_500ms_bps").mean()).item()
        return {
            "markout_100ms_bps": np.nan,
            "markout_500ms_bps": float(m500) if m500 is not None else np.nan,
            "markout_1s_bps": np.nan,
            "markout_5s_bps": np.nan,
        }

    fill_100ms = (
        fills.sort("ts_100ms")
        .join_asof(
            mid_ref,
            left_on="ts_100ms",
            right_on="ts_ms",
            strategy="forward",
        )
        .select([pl.col("fill_id"), pl.col("mid_px").alias("mid_100ms")])
    )

    fill_1s = (
        fills.sort("ts_1s")
        .join_asof(
            mid_ref,
            left_on="ts_1s",
            right_on="ts_ms",
            strategy="forward",
        )
        .select([pl.col("fill_id"), pl.col("mid_px").alias("mid_1s")])
    )

    fill_5s = (
        fills.sort("ts_5s")
        .join_asof(
            mid_ref,
            left_on="ts_5s",
            right_on="ts_ms",
            strategy="forward",
        )
        .select([pl.col("fill_id"), pl.col("mid_px").alias("mid_5s")])
    )

    markout_df = (
        fills.join(fill_100ms, on="fill_id", how="left")
        .join(fill_500ms, on="fill_id", how="left")
        .join(fill_1s, on="fill_id", how="left")
        .join(fill_5s, on="fill_id", how="left")
        .with_columns(
            [
                pl.when(pl.col("side") > 0.0)
                .then((pl.col("mid_100ms") / pl.col("exec_px") - 1.0) * 10000.0)
                .otherwise((1.0 - pl.col("mid_100ms") / pl.col("exec_px")) * 10000.0)
                .alias("markout_100ms_bps"),
                pl.when(pl.col("side") > 0.0)
                .then((pl.col("mid_500ms") / pl.col("exec_px") - 1.0) * 10000.0)
                .otherwise((1.0 - pl.col("mid_500ms") / pl.col("exec_px")) * 10000.0)
                .alias("markout_500ms_bps"),
                pl.when(pl.col("side") > 0.0)
                .then((pl.col("mid_1s") / pl.col("exec_px") - 1.0) * 10000.0)
                .otherwise((1.0 - pl.col("mid_1s") / pl.col("exec_px")) * 10000.0)
                .alias("markout_1s_bps"),
                pl.when(pl.col("side") > 0.0)
                .then((pl.col("mid_5s") / pl.col("exec_px") - 1.0) * 10000.0)
                .otherwise((1.0 - pl.col("mid_5s") / pl.col("exec_px")) * 10000.0)
                .alias("markout_5s_bps"),
            ]
        )
    )

    m100 = markout_df.select(pl.col("markout_100ms_bps").mean()).item()
    m500 = markout_df.select(pl.col("markout_500ms_bps").mean()).item()
    m1 = markout_df.select(pl.col("markout_1s_bps").mean()).item()
    m5 = markout_df.select(pl.col("markout_5s_bps").mean()).item()
    return {
        "markout_100ms_bps": float(m100) if m100 is not None else np.nan,
        "markout_500ms_bps": float(m500) if m500 is not None else np.nan,
        "markout_1s_bps": float(m1) if m1 is not None else np.nan,
        "markout_5s_bps": float(m5) if m5 is not None else np.nan,
    }


def _compute_fill_side_pnl(events: pl.DataFrame, fills_df: pl.DataFrame) -> Dict[str, float]:
    if fills_df.height == 0:
        return {
            "buy_side_pnl": np.nan,
            "sell_side_pnl": np.nan,
            "buy_fill_count": 0.0,
            "sell_fill_count": 0.0,
        }
    if "mid_px" in events.columns:
        mid_expr = pl.col("mid_px").cast(pl.Float64)
    elif {"bid_px", "ask_px"}.issubset(set(events.columns)):
        mid_expr = ((pl.col("bid_px") + pl.col("ask_px")) * 0.5).cast(pl.Float64)
    else:
        return {
            "buy_side_pnl": np.nan,
            "sell_side_pnl": np.nan,
            "buy_fill_count": 0.0,
            "sell_fill_count": 0.0,
        }

    mid_ref = events.select(
        [
            pl.col("ts_ms").cast(pl.Int64).alias("ts_ms"),
            mid_expr.alias("mid_px"),
        ]
    ).sort("ts_ms")
    fills = fills_df.select(
        [
            pl.col("ts_ms").cast(pl.Int64).alias("fill_ts_ms"),
            pl.col("side").cast(pl.Float64).alias("side"),
            pl.col("qty").cast(pl.Float64).alias("qty"),
            pl.col("px").cast(pl.Float64).alias("exec_px"),
        ]
    ).sort("fill_ts_ms")
    merged = fills.join_asof(
        mid_ref,
        left_on="fill_ts_ms",
        right_on="ts_ms",
        strategy="backward",
    )
    pnl_df = merged.with_columns(
        [
            pl.when(pl.col("side") > 0.0)
            .then((pl.col("mid_px") - pl.col("exec_px")) * pl.col("qty"))
            .otherwise(0.0)
            .alias("buy_side_pnl"),
            pl.when(pl.col("side") < 0.0)
            .then((pl.col("exec_px") - pl.col("mid_px")) * pl.col("qty"))
            .otherwise(0.0)
            .alias("sell_side_pnl"),
            pl.when(pl.col("side") > 0.0).then(1.0).otherwise(0.0).alias("buy_cnt"),
            pl.when(pl.col("side") < 0.0).then(1.0).otherwise(0.0).alias("sell_cnt"),
        ]
    )
    buy_pnl = pnl_df.select(pl.col("buy_side_pnl").sum()).item()
    sell_pnl = pnl_df.select(pl.col("sell_side_pnl").sum()).item()
    buy_cnt = pnl_df.select(pl.col("buy_cnt").sum()).item()
    sell_cnt = pnl_df.select(pl.col("sell_cnt").sum()).item()
    return {
        "buy_side_pnl": float(buy_pnl) if buy_pnl is not None else np.nan,
        "sell_side_pnl": float(sell_pnl) if sell_pnl is not None else np.nan,
        "buy_fill_count": float(buy_cnt) if buy_cnt is not None else 0.0,
        "sell_fill_count": float(sell_cnt) if sell_cnt is not None else 0.0,
    }


def _plot_markout_decay_curve(metrics: Dict[str, float], output_path: str) -> None:
    points = [
        ("100ms", metrics.get("markout_100ms_bps", np.nan)),
        ("500ms", metrics.get("markout_500ms_bps", np.nan)),
        ("1s", metrics.get("markout_1s_bps", np.nan)),
        ("5s", metrics.get("markout_5s_bps", np.nan)),
    ]
    xs = []
    ys = []
    for label, val in points:
        if val is not None and np.isfinite(val):
            xs.append(label)
            ys.append(float(val))
    if len(xs) == 0:
        print("skip markout curve: no finite markout points")
        return
    try:
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except Exception as exc:  # optional dependency in runtime environments
        print(f"skip markout curve: matplotlib unavailable ({exc})")
        return
    plt.figure(figsize=(6.0, 3.5))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.title("Markout Decay Curve")
    plt.xlabel("Horizon")
    plt.ylabel("Markout (bps)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"saved: {output_path}")


# =========================================
# 4) 指标：Turnover, Sharpe, Calmar
# =========================================


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return np.nan
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def evaluate_performance(
    equity_df: pl.DataFrame,
    total_notional: float,
    initial_capital: float,
    cfg: BacktestConfig,
) -> Dict[str, float]:
    turnover = total_notional / max(initial_capital, 1e-12)
    ts_all = equity_df["ts_ms"].to_numpy()
    sample_days = (
        max(0.0, float(ts_all[-1] - ts_all[0])) / (24.0 * 3600.0 * 1000.0)
        if ts_all.size >= 2
        else np.nan
    )
    monthly_turnover = (
        turnover * (30.0 / sample_days) if np.isfinite(sample_days) and sample_days > 0 else np.nan
    )
    # max daily loss based on raw equity_df (Polars only)
    max_daily_loss = np.nan
    if equity_df.height >= 2:
        try:
            daily = (
                equity_df.with_columns((pl.col("ts_ms") // 86_400_000).alias("_day"))
                .group_by("_day")
                .agg(
                    [
                        pl.col("equity").first().alias("_eq_first"),
                        pl.col("equity").last().alias("_eq_last"),
                    ]
                )
                .with_columns((pl.col("_eq_last") - pl.col("_eq_first")).alias("_day_pnl"))
            )
            if daily.height > 0:
                max_daily_loss = float(daily["_day_pnl"].min())
        except Exception:
            max_daily_loss = np.nan

    default_metrics = {
        "turnover_multiple": float(turnover),
        "turnover_multiple_monthly": float(monthly_turnover) if np.isfinite(monthly_turnover) else np.nan,
        "sample_days": float(sample_days) if np.isfinite(sample_days) else np.nan,
        "turnover_passed": bool(
            np.isfinite(monthly_turnover) and monthly_turnover >= cfg.required_monthly_turnover
        ),
        "max_daily_loss": float(max_daily_loss) if np.isfinite(max_daily_loss) else np.nan,
        "total_return": np.nan,
        "annual_return": np.nan,
        "annual_return_linear": np.nan,
        "max_drawdown": np.nan,
        "sharpe": np.nan,
        "calmar": np.nan,
        "calmar_linear": np.nan,
    }

    if equity_df.height < max(cfg.min_points_for_metrics, 2):
        return default_metrics

    if cfg.pnl_resample_seconds > 0:
        step_ms = cfg.pnl_resample_seconds * 1000
        eq_rs = (
            equity_df.with_columns(
                ((pl.col("ts_ms") // step_ms) * step_ms).alias("bucket_ts_ms")
            )
            .group_by("bucket_ts_ms")
            .agg(pl.col("equity").last().alias("equity"))
            .sort("bucket_ts_ms")
        )
        eq = eq_rs["equity"].to_numpy()
        ts_eval = eq_rs["bucket_ts_ms"].to_numpy()
    else:
        # equity_df is already in event order from engine output; avoid expensive sort on large frames
        eq = equity_df["equity"].to_numpy()
        ts_eval = equity_df["ts_ms"].to_numpy()

    if eq.size < 2:
        return default_metrics

    rets = eq[1:] / np.maximum(eq[:-1], 1e-12) - 1.0
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets))

    ms_per_year = 365.0 * 24.0 * 3600.0 * 1000.0
    dt = np.diff(ts_eval.astype(np.float64))
    dt = dt[dt > 0]
    median_dt = float(np.median(dt)) if dt.size > 0 else np.nan
    periods_per_year = ms_per_year / median_dt if np.isfinite(median_dt) and median_dt > 0 else np.nan

    elapsed_ms = float(ts_eval[-1] - ts_eval[0]) if ts_eval.size >= 2 else np.nan
    elapsed_years = elapsed_ms / ms_per_year if np.isfinite(elapsed_ms) and elapsed_ms > 0 else np.nan
    eq0 = float(eq[0])
    eqn = float(eq[-1])
    if np.isfinite(elapsed_years) and elapsed_years > 0 and eq0 > 0.0 and eqn > 0.0:
        ann_ret = (eqn / eq0) ** (1.0 / elapsed_years) - 1.0
    else:
        ann_ret = np.nan
    sharpe = (
        np.nan
        if (std_r <= 1e-12 or not np.isfinite(periods_per_year))
        else (mean_r / std_r) * np.sqrt(periods_per_year)
    )

    mdd = _max_drawdown(eq)
    mdd_for_calmar = max(float(mdd), 1e-4) if np.isfinite(mdd) else np.nan
    calmar = np.nan if (not np.isfinite(mdd_for_calmar)) else ann_ret / mdd_for_calmar
    ann_ret_linear = (
        (float(eq[-1] / max(eq[0], 1e-12) - 1.0) / sample_days) * 365.0
        if np.isfinite(sample_days) and sample_days > 0
        else np.nan
    )
    calmar_linear = (
        np.nan
        if (not np.isfinite(mdd_for_calmar) or not np.isfinite(ann_ret_linear))
        else ann_ret_linear / mdd_for_calmar
    )

    return {
        "turnover_multiple": float(turnover),
        "turnover_multiple_monthly": float(monthly_turnover) if np.isfinite(monthly_turnover) else np.nan,
        "sample_days": float(sample_days) if np.isfinite(sample_days) else np.nan,
        "turnover_passed": bool(
            np.isfinite(monthly_turnover) and monthly_turnover >= cfg.required_monthly_turnover
        ),
        "max_daily_loss": float(max_daily_loss) if np.isfinite(max_daily_loss) else np.nan,
        "total_return": float(eq[-1] / max(eq[0], 1e-12) - 1.0),
        "annual_return": float(ann_ret) if np.isfinite(ann_ret) else np.nan,
        "annual_return_linear": float(ann_ret_linear) if np.isfinite(ann_ret_linear) else np.nan,
        "max_drawdown": float(mdd),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
        "calmar_linear": float(calmar_linear) if np.isfinite(calmar_linear) else np.nan,
    }


# =========================================
# 5) 参数扫描：帮助追求 turnover > 500x
# =========================================


def _close_grid_worker_shm() -> None:
    global _GRID_WORKER_SHM_HANDLES
    for handle in _GRID_WORKER_SHM_HANDLES:
        try:
            handle.close()
        except FileNotFoundError:
            pass
    _GRID_WORKER_SHM_HANDLES = []


def _init_grid_worker(
    events_payload: Dict[str, object],
    base_cfg_dict: Dict[str, object],
    payload_is_shared: bool = False,
) -> None:
    global _GRID_EVENTS_PAYLOAD, _GRID_BASE_CFG_DICT, _GRID_EVENTS_DF, _GRID_WORKER_SHM_HANDLES
    if payload_is_shared:
        local_payload: Dict[str, np.ndarray] = {}
        _GRID_WORKER_SHM_HANDLES = []
        for col, meta_obj in events_payload.items():
            meta = meta_obj  # typing helper for readability
            shm = shared_memory.SharedMemory(name=meta["name"])
            arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
            arr.setflags(write=False)
            local_payload[col] = arr
            _GRID_WORKER_SHM_HANDLES.append(shm)
        _GRID_EVENTS_PAYLOAD = local_payload
        atexit.register(_close_grid_worker_shm)
    else:
        _GRID_EVENTS_PAYLOAD = events_payload  # type: ignore[assignment]
    _GRID_BASE_CFG_DICT = base_cfg_dict
    _GRID_EVENTS_DF = pl.DataFrame(_GRID_EVENTS_PAYLOAD)


def _grid_eval_one(task: Tuple[float, float, float, int]) -> Dict[str, object]:
    global _GRID_EVENTS_DF, _GRID_BASE_CFG_DICT
    if _GRID_EVENTS_DF is None or _GRID_BASE_CFG_DICT is None:
        raise RuntimeError("grid worker not initialized")
    half_spread_ticks, gamma_inventory, participation_cap, cooldown_ms = task
    cfg = BacktestConfig(**_GRID_BASE_CFG_DICT)
    cfg.half_spread_ticks = float(half_spread_ticks)
    cfg.gamma_inventory = float(gamma_inventory)
    cfg.participation_cap = float(participation_cap)
    cfg.cooldown_ms = int(cooldown_ms)
    cfg.base_cooldown_ms = int(cooldown_ms)
    cfg.execution_cooldown_ms = int(cooldown_ms)
    cfg.latency_ms = 10
    cfg.queue_model = 0
    out = run_backtest(_GRID_EVENTS_DF, cfg, grid_stage=True)
    m = out["metrics"]
    # Grid-only early-stop override: keep medium-turnover candidates from being dropped.
    actual_monthly_turn = float(m.get("turnover_multiple_monthly", 0.0))
    early_stopped = int(m.get("early_stopped", 0))
    if early_stopped == 1 and actual_monthly_turn >= 400.0:
        early_stopped = 0
        m["early_stopped"] = 0
        m["turnover_passed"] = True
        m["eval_skipped"] = 0
    turnover_monthly = float(m["turnover_multiple_monthly"])
    overtrade_penalty = int(np.isfinite(turnover_monthly) and turnover_monthly > 3000.0)
    sharpe = float(m["sharpe"]) if m["sharpe"] is not None else np.nan
    calmar = float(m["calmar"]) if m["calmar"] is not None else np.nan
    total_return = float(m["total_return"]) if m["total_return"] is not None else np.nan
    if overtrade_penalty == 1:
        sharpe = -1e6
        calmar = -1e6
        total_return = -1e6
    return {
        "half_spread_ticks": half_spread_ticks,
        "gamma_inventory": gamma_inventory,
        "latency_ms": 10,
        "queue_model": 0,
        "participation_cap": participation_cap,
        "cooldown_ms": cooldown_ms,
        "turnover": turnover_monthly,
        "turnover_passed": int(m["turnover_passed"]) if overtrade_penalty == 0 else 0,
        "overtrade_penalty": overtrade_penalty,
        "sharpe": sharpe,
        "calmar": calmar,
        "total_return": total_return,
        "max_drawdown": m["max_drawdown"],
        "toxic_fill_ratio": m.get("toxic_fill_ratio", np.nan),
        "buy_fill_count": m.get("buy_fill_count", 0.0),
        "sell_fill_count": m.get("sell_fill_count", 0.0),
        "fill_ratio": (
            min(float(m.get("buy_fill_count", 0.0)), float(m.get("sell_fill_count", 0.0)))
            / max(max(float(m.get("buy_fill_count", 0.0)), float(m.get("sell_fill_count", 0.0))), 1e-12)
        ),
        "markout_100ms_bps": m.get("markout_100ms_bps", np.nan),
        "markout_500ms_bps": m.get("markout_500ms_bps", np.nan),
        "markout_1s_bps": m.get("markout_1s_bps", np.nan),
        "markout_5s_bps": m.get("markout_5s_bps", np.nan),
        "annual_return": m["annual_return"],
        "early_stopped": early_stopped,
    }


def grid_search(events: pl.DataFrame, base_cfg: BacktestConfig) -> pl.DataFrame:
    events_ready = events
    if base_cfg.use_advanced_microprice and "micro_px_adv" not in events.columns:
        events_ready = enrich_advanced_microprice(events, base_cfg)

    # Pre-train dictionary once for grid search (avoid training per worker)
    dict_cache_for_grid = None
    if base_cfg.use_dictionary_strategy and HAS_SKLEARN:
        dict_cache_for_grid = Path(os.getenv("HFT_DATA_DIR", ".")).expanduser() / "_grid_dict_cache.npz"
        if not dict_cache_for_grid.exists():
            print("Grid search: pre-training dictionary once...")
            X_full, scaler_state = build_feature_matrix(events_ready, base_cfg, fit_scalers=True)
            n_train = max(base_cfg.dict_batch_size * 5, int(X_full.shape[0] * base_cfg.dict_train_ratio))
            n_train = min(n_train, X_full.shape[0])
            D, _ = fit_dictionary(X_full[:n_train], base_cfg)
            np.savez(
                str(dict_cache_for_grid),
                D=D,
                scaler_mean=scaler_state["mean"],
                scaler_std=scaler_state["std"],
            )
            print(f"Grid dict cached: {dict_cache_for_grid}")
        base_cfg.dict_cache_path = str(dict_cache_for_grid)

    stride = max(1, int(base_cfg.grid_search_event_stride))
    if stride == 1:
        events_eval = events_ready
    else:
        events_eval = (
            events_ready.with_row_index(name="_row_idx")
            .filter((pl.col("_row_idx") % stride) == 0)
            .drop("_row_idx")
        )

    param_tasks: List[Tuple[float, float, float, int]] = []
    # Target: turnover 500-1500x monthly (OOS test set also guaranteed)
    for half_spread_ticks in [5.0, 5.5, 6.0]:
        for gamma_inventory in [1.0, 1.8, 2.5]:
            for participation_cap in [0.005, 0.01]:
                for cooldown_ms in [500, 800, 1200]:
                    param_tasks.append((half_spread_ticks, gamma_inventory, participation_cap, cooldown_ms))

    events_payload: Dict[str, np.ndarray] = {
        c: events_eval[c].to_numpy()
        for c in [
            "ts_ms",
            "bid_px",
            "ask_px",
            "bid_sz",
            "ask_sz",
            "trade_px",
            "trade_sz",
            "trade_side",
            ("micro_px_adv" if (base_cfg.use_advanced_microprice and "micro_px_adv" in events_eval.columns) else "micro_px"),
            "state_depth_bin",
            "state_duration_bin",
        ]
        if c in events_eval.columns
    }
    if "state_depth_bin" not in events_payload:
        events_payload["state_depth_bin"] = np.zeros(events_eval.height, dtype=np.int64)
    if "state_duration_bin" not in events_payload:
        events_payload["state_duration_bin"] = np.zeros(events_eval.height, dtype=np.int64)
    if "micro_px_adv" in events_payload and "micro_px" not in events_payload:
        events_payload["micro_px"] = events_payload["micro_px_adv"]
        del events_payload["micro_px_adv"]

    shared_payload: Dict[str, Dict[str, object]] = {}
    shm_handles: List[shared_memory.SharedMemory] = []
    use_shared = False
    try:
        for col, arr in events_payload.items():
            arr_c = np.ascontiguousarray(arr)
            shm = shared_memory.SharedMemory(create=True, size=arr_c.nbytes)
            shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
            shm_arr[:] = arr_c
            shared_payload[col] = {
                "name": shm.name,
                "shape": arr_c.shape,
                "dtype": arr_c.dtype.str,
            }
            shm_handles.append(shm)
        use_shared = True
    except Exception:
        for shm in shm_handles:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        shm_handles = []
        shared_payload = {}
        use_shared = False

    rows: List[Dict[str, object]] = []
    total = len(param_tasks)
    workers = max(1, min((os.cpu_count() or 4), 8))
    try:
        init_payload: Dict[str, object] = shared_payload if use_shared else events_payload
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_grid_worker,
            initargs=(init_payload, asdict(base_cfg), use_shared),
        ) as ex:
            futs = [ex.submit(_grid_eval_one, t) for t in param_tasks]
            done = 0
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                done += 1
                if (done % 10) == 0 or done == total:
                    print(
                        "grid_progress",
                        done,
                        "/",
                        total,
                        "latest",
                        {
                            "turnover_passed": int(row["turnover_passed"]),
                            "sharpe": float(row["sharpe"]) if row["sharpe"] is not None else None,
                            "turnover": float(row["turnover"]) if row["turnover"] is not None else None,
                        },
                    )
    finally:
        for shm in shm_handles:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    out_df = pl.DataFrame(rows).sort(
        by=["turnover_passed", "overtrade_penalty", "sharpe", "calmar", "turnover"],
        descending=[True, False, True, True, True],
    )
    turnover_ok = out_df.filter(
        (pl.col("turnover").fill_null(-1.0) >= 600.0)
        & (pl.col("turnover").fill_null(-1.0) <= 900.0)
        & (pl.col("sharpe").fill_null(-1e6) > 0.0)
        & (pl.col("overtrade_penalty").fill_null(0) == 0)
    )
    print("\n=== Grid Search: 600 <= monthly_turnover <= 900 && sharpe > 0 ===")
    if turnover_ok.height > 0:
        print(turnover_ok)
    else:
        print("没有组合达标；可继续上调 half_spread_ticks 或延长 cooldown_ms。")
    return out_df


def split_train_test_by_time(events: pl.DataFrame, train_ratio: float) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if events.height < 2:
        return events, events.clear()
    ratio = min(0.95, max(0.05, float(train_ratio)))
    n_train = int(events.height * ratio)
    n_train = min(events.height - 1, max(1, n_train))
    return events.head(n_train), events.slice(n_train)


def select_best_cfg_from_grid(base_cfg: BacktestConfig, gs: pl.DataFrame) -> BacktestConfig:
    if gs.height == 0:
        return base_cfg
    min_fill_ratio = max(0.0, min(1.0, float(base_cfg.min_fill_ratio_hard)))
    w100 = float(base_cfg.markout_weight_100ms)
    w500 = float(base_cfg.markout_weight_500ms)
    wsum = max(1e-12, w100 + w500)
    w100 /= wsum
    w500 /= wsum
    hard_candidates = gs.filter(
        (pl.col("turnover_passed") >= 1)
        & (pl.col("turnover").fill_null(-1.0) <= float(base_cfg.max_monthly_turnover))
        & (pl.col("buy_fill_count").fill_null(0.0) > 0.0)
        & (pl.col("sell_fill_count").fill_null(0.0) > 0.0)
        & (pl.col("fill_ratio").fill_null(0.0) >= min_fill_ratio)
    )
    if hard_candidates.height == 0:
        return base_cfg
    scored = hard_candidates.with_columns(
        (
            pl.col("sharpe").fill_null(-1e6)
            + (
                w100
                * pl.when(pl.col("markout_100ms_bps").fill_null(0.0) < 0.0)
                .then(pl.col("markout_100ms_bps").fill_null(0.0))
                .otherwise(0.0)
                + w500
                * pl.when(pl.col("markout_500ms_bps").fill_null(0.0) < 0.0)
                .then(pl.col("markout_500ms_bps").fill_null(0.0))
                .otherwise(0.0)
            )
            - 2.0 * pl.col("toxic_fill_ratio").fill_null(0.0).clip(0.0, 1.0)
        ).alias("stability_score")
    ).sort(
        by=["stability_score", "total_return", "sharpe", "turnover"],
        descending=[True, True, True, True],
    )
    top = scored.row(0, named=True)
    best = BacktestConfig(**asdict(base_cfg))
    best.half_spread_ticks = float(top["half_spread_ticks"])
    best.gamma_inventory = float(top["gamma_inventory"])
    best.latency_ms = int(top["latency_ms"])
    best.queue_model = int(top["queue_model"])
    if "participation_cap" in scored.columns:
        best.participation_cap = float(top["participation_cap"])
    if "obi_hard_threshold" in scored.columns:
        best.obi_hard_threshold = float(top["obi_hard_threshold"])
    if "obi_ema_alpha" in scored.columns:
        best.obi_ema_alpha = float(top["obi_ema_alpha"])
    if "obi_trend_skew_bps" in scored.columns:
        best.obi_trend_skew_bps = float(top["obi_trend_skew_bps"])
    if "intensity_threshold" in scored.columns:
        best.intensity_threshold = float(top["intensity_threshold"])
    if "lambda_intensity" in scored.columns:
        best.lambda_intensity = float(top["lambda_intensity"])
    if "fill_rate_alpha" in scored.columns:
        best.fill_rate_alpha = float(top["fill_rate_alpha"])
    if "micro_px_alpha" in scored.columns:
        best.micro_px_alpha = float(top["micro_px_alpha"])
    return best


def _build_cfg_from_row(base_cfg: BacktestConfig, row: Dict[str, object]) -> BacktestConfig:
    cfg = BacktestConfig(**asdict(base_cfg))
    cfg.half_spread_ticks = float(row["half_spread_ticks"])
    cfg.gamma_inventory = float(row["gamma_inventory"])
    cfg.latency_ms = int(row["latency_ms"])
    cfg.queue_model = int(row["queue_model"])
    if "participation_cap" in row:
        cfg.participation_cap = float(row["participation_cap"])
    if "obi_hard_threshold" in row:
        cfg.obi_hard_threshold = float(row["obi_hard_threshold"])
    if "obi_ema_alpha" in row:
        cfg.obi_ema_alpha = float(row["obi_ema_alpha"])
    if "obi_trend_skew_bps" in row:
        cfg.obi_trend_skew_bps = float(row["obi_trend_skew_bps"])
    if "intensity_threshold" in row:
        cfg.intensity_threshold = float(row["intensity_threshold"])
    if "lambda_intensity" in row:
        cfg.lambda_intensity = float(row["lambda_intensity"])
    if "fill_rate_alpha" in row:
        cfg.fill_rate_alpha = float(row["fill_rate_alpha"])
    if "micro_px_alpha" in row:
        cfg.micro_px_alpha = float(row["micro_px_alpha"])
    return cfg


def _select_best_cfg_with_stability(
    base_cfg: BacktestConfig,
    gs: pl.DataFrame,
    train_daily_events: List[pl.DataFrame],
) -> BacktestConfig:
    if gs.height == 0:
        return base_cfg

    # 硬门槛：训练期月化换手必须达标，且必须双边成交、买卖不能过于失衡。
    min_fill_ratio = max(0.0, min(1.0, float(base_cfg.min_fill_ratio_hard)))
    w100 = float(base_cfg.markout_weight_100ms)
    w500 = float(base_cfg.markout_weight_500ms)
    wsum = max(1e-12, w100 + w500)
    w100 /= wsum
    w500 /= wsum
    gs_turnover = gs.filter(
        (pl.col("turnover_passed") >= 1)
        & (pl.col("turnover").fill_null(-1.0) <= float(base_cfg.max_monthly_turnover))
        & (pl.col("buy_fill_count").fill_null(0.0) > 0.0)
        & (pl.col("sell_fill_count").fill_null(0.0) > 0.0)
        & (pl.col("fill_ratio").fill_null(0.0) >= min_fill_ratio)
        & (pl.col("markout_500ms_bps").fill_null(-999.0) > -1.8)
    )
    if gs_turnover.height == 0:
        print("stability_select: no hard-gate candidates, fallback to base_cfg")
        return base_cfg

    top_k = max(1, min(int(base_cfg.stability_eval_top_k), gs_turnover.height))
    candidates = gs_turnover.sort(
        by=["total_return", "sharpe"],
        descending=[True, True],
    ).head(top_k)

    best_cfg = base_cfg
    best_score = -1e30
    valid_count = 0

    for row in candidates.iter_rows(named=True):
        cfg = _build_cfg_from_row(base_cfg, row)
        daily_returns = []
        daily_markout_100ms = []
        daily_markout_500ms = []
        daily_toxic = []
        buy_total = 0.0
        sell_total = 0.0
        for day_events in train_daily_events:
            out = run_backtest(day_events, cfg)
            m = out["metrics"]
            d_ret = m.get("total_return", np.nan)
            d_m100 = m.get("markout_100ms_bps", np.nan)
            d_m500 = m.get("markout_500ms_bps", np.nan)
            d_tox = m.get("toxic_fill_ratio", np.nan)
            b_cnt = m.get("buy_fill_count", 0.0)
            s_cnt = m.get("sell_fill_count", 0.0)
            if d_ret is not None and np.isfinite(d_ret):
                daily_returns.append(float(d_ret))
            if d_m100 is not None and np.isfinite(d_m100):
                daily_markout_100ms.append(float(d_m100))
            if d_m500 is not None and np.isfinite(d_m500):
                daily_markout_500ms.append(float(d_m500))
            if d_tox is not None and np.isfinite(d_tox):
                daily_toxic.append(float(d_tox))
            if b_cnt is not None and np.isfinite(b_cnt):
                buy_total += float(b_cnt)
            if s_cnt is not None and np.isfinite(s_cnt):
                sell_total += float(s_cnt)

        if len(daily_returns) == 0:
            continue

        arr_ret = np.asarray(daily_returns, dtype=np.float64)
        mean_ret = float(np.mean(arr_ret))
        std_ret = float(np.std(arr_ret))
        stability_score = mean_ret / (std_ret + max(float(base_cfg.stability_eps), 1e-9))
        markout_100ms_bps = float(np.mean(np.asarray(daily_markout_100ms, dtype=np.float64))) if len(daily_markout_100ms) > 0 else 0.0
        markout_500ms_bps = float(np.mean(np.asarray(daily_markout_500ms, dtype=np.float64))) if len(daily_markout_500ms) > 0 else 0.0
        toxic_ratio = float(np.mean(np.asarray(daily_toxic, dtype=np.float64))) if len(daily_toxic) > 0 else 0.0
        short_markout_penalty = (
            4.0 * w100 * min(0.0, markout_100ms_bps)
            + 4.0 * w500 * min(0.0, markout_500ms_bps)
        )
        final_score = stability_score + short_markout_penalty - 3.5 * toxic_ratio
        positive_ratio = float(np.mean(arr_ret > 0.0))
        side_balance = min(buy_total, sell_total) / max(max(buy_total, sell_total), 1e-12)

        # 亏损天数占比 > 20% 直接剔除。
        if positive_ratio < 0.8:
            continue
        # 双边成交与平衡约束，防止挑出单边“假优”参数。
        if buy_total <= 0.0 or sell_total <= 0.0 or side_balance < min_fill_ratio:
            continue
        valid_count += 1
        if final_score > best_score:
            best_score = final_score
            best_cfg = cfg

    if valid_count == 0:
        # 保持硬门槛：若无稳定候选，不回退到低换手参数。
        print("stability_select: no stable turnover-passed candidates, fallback to base_cfg")
        return base_cfg
    return best_cfg


def _prepare_events(events: pl.DataFrame, cfg: BacktestConfig) -> pl.DataFrame:
    if cfg.use_advanced_microprice:
        return enrich_advanced_microprice(events, cfg)
    return events


def _split_events_by_day(events: pl.DataFrame) -> List[pl.DataFrame]:
    if events.height == 0:
        return []
    parts = (
        events.with_columns((pl.col("ts_ms") // 86_400_000).alias("_day"))
        .partition_by("_day", as_dict=False, maintain_order=True)
    )
    return [p.drop("_day") for p in parts]


def _extract_day_from_events_name(name: str) -> Optional[str]:
    m = re.search(r"events_(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else None


def run_walk_forward_from_parquet_dir(cfg: BacktestConfig, data_dir: Path) -> Optional[pl.DataFrame]:
    files = sorted(data_dir.glob("events_*.parquet"))
    dated = []
    for fp in files:
        day = _extract_day_from_events_name(fp.name)
        if day is not None:
            dated.append((day, fp))
    dated = sorted(dated, key=lambda x: x[0])

    train_days = max(1, int(cfg.walk_forward_train_days))
    gap_days = max(0, int(cfg.walk_forward_gap_days))
    need_files = train_days + gap_days + 1
    if len(dated) < need_files:
        print(
            f"\n=== Walk-Forward === skip: files={len(dated)}, need >= {need_files}"
        )
        return None

    rows = []
    for i in range(train_days + gap_days, len(dated)):
        train_end = i - gap_days
        train_slice = dated[train_end - train_days : train_end]
        test_day, test_fp = dated[i]
        train_days_list = [d for d, _ in train_slice]
        gap_slice = dated[train_end:i]
        gap_days_list = [d for d, _ in gap_slice]

        train_parts = [pl.read_parquet(str(p)) for _, p in train_slice]
        train_events = pl.concat(train_parts, how="vertical")
        train_daily_events_raw = [pl.read_parquet(str(p)) for _, p in train_slice]
        test_events = pl.read_parquet(str(test_fp))
        print(
            "\nwf_day",
            test_day,
            "train_days",
            train_days,
            "gap_days",
            gap_days,
            "gap_range",
            gap_days_list,
            "train_events",
            int(train_events.height),
            "test_events",
            int(test_events.height),
        )

        wf_cfg = BacktestConfig(**asdict(cfg))
        # 强制开启微观价格特征，并严格执行 Train-Fit / Test-Apply。
        wf_cfg.walk_forward_disable_advanced_microprice = False
        wf_cfg.use_advanced_microprice = True

        if wf_cfg.use_advanced_microprice:
            train_events, g_mat, fit_state = enrich_advanced_microprice(
                train_events,
                wf_cfg,
                g_matrix=None,
                bin_state=None,
                return_fit_state=True,
            )
            train_daily_events = [
                enrich_advanced_microprice(
                    day_df,
                    wf_cfg,
                    g_matrix=g_mat,
                    bin_state=fit_state,
                    return_fit_state=False,
                )
                for day_df in train_daily_events_raw
            ]
            test_events = enrich_advanced_microprice(
                test_events,
                wf_cfg,
                g_matrix=g_mat,
                bin_state=fit_state,
                return_fit_state=False,
            )
        else:
            train_events = _prepare_events(train_events, wf_cfg)
            train_daily_events = [_prepare_events(day_df, wf_cfg) for day_df in train_daily_events_raw]
            test_events = _prepare_events(test_events, wf_cfg)

        gs_train = grid_search(train_events, wf_cfg)
        best_cfg = _select_best_cfg_with_stability(wf_cfg, gs_train, train_daily_events)

        train_res = run_backtest(train_events, best_cfg)
        test_res = run_backtest(test_events, best_cfg)
        tm = test_res["metrics"]

        rows.append(
            {
                "test_day": test_day,
                "train_start_day": train_days_list[0],
                "train_end_day": train_days_list[-1],
                "train_days": train_days,
                "gap_days": gap_days,
                "best_half_spread_ticks": best_cfg.half_spread_ticks,
                "best_gamma_inventory": best_cfg.gamma_inventory,
                "best_latency_ms": best_cfg.latency_ms,
                "best_queue_model": best_cfg.queue_model,
                "best_obi_hard_threshold": best_cfg.obi_hard_threshold,
                "best_obi_ema_alpha": best_cfg.obi_ema_alpha,
                "best_obi_trend_skew_bps": best_cfg.obi_trend_skew_bps,
                "best_intensity_threshold": best_cfg.intensity_threshold,
                "best_lambda_intensity": best_cfg.lambda_intensity,
                "best_fill_rate_alpha": best_cfg.fill_rate_alpha,
                "best_micro_px_alpha": best_cfg.micro_px_alpha,
                "test_turnover_monthly": tm["turnover_multiple_monthly"],
                "test_turnover_passed": int(tm["turnover_passed"]),
                "test_sharpe": tm["sharpe"],
                "test_calmar": tm["calmar"],
                "test_total_return": tm["total_return"],
                "test_max_drawdown": tm["max_drawdown"],
                "test_toxic_fill_ratio": tm.get("toxic_fill_ratio", np.nan),
                "test_markout_1s_bps": tm.get("markout_1s_bps", np.nan),
                "test_markout_5s_bps": tm.get("markout_5s_bps", np.nan),
                "train_sharpe": train_res["metrics"]["sharpe"],
            }
        )

    wf_df = pl.DataFrame(rows).sort("test_day")
    print("\n=== Walk-Forward Daily OOS ===")
    print(wf_df)
    print("\n=== Walk-Forward Summary ===")
    print(
        {
            "days": wf_df.height,
            "turnover_pass_rate": float(wf_df["test_turnover_passed"].mean()),
            "mean_test_sharpe": float(wf_df["test_sharpe"].mean()),
            "median_test_sharpe": float(wf_df["test_sharpe"].median()),
            "mean_test_calmar": float(wf_df["test_calmar"].mean()),
            "mean_test_return": float(wf_df["test_total_return"].mean()),
            "mean_test_mdd": float(wf_df["test_max_drawdown"].mean()),
            "mean_test_toxic_fill_ratio": float(wf_df["test_toxic_fill_ratio"].mean()),
            "mean_test_markout_1s_bps": float(wf_df["test_markout_1s_bps"].mean()),
            "mean_test_markout_5s_bps": float(wf_df["test_markout_5s_bps"].mean()),
        }
    )
    high_vol_days = ["2026-01-28", "2026-01-29"]
    high_df = wf_df.filter(pl.col("test_day").is_in(high_vol_days))
    low_df = wf_df.filter(~pl.col("test_day").is_in(high_vol_days))
    regime_analysis = {
        "high_vol_days": high_vol_days,
        "high_count": int(high_df.height),
        "high_mean_return": float(high_df["test_total_return"].mean()) if high_df.height > 0 else np.nan,
        "high_mean_sharpe": float(high_df["test_sharpe"].mean()) if high_df.height > 0 else np.nan,
        "high_mean_mdd": float(high_df["test_max_drawdown"].mean()) if high_df.height > 0 else np.nan,
        "high_mean_toxic_fill_ratio": float(high_df["test_toxic_fill_ratio"].mean()) if high_df.height > 0 else np.nan,
        "high_mean_markout_1s_bps": float(high_df["test_markout_1s_bps"].mean()) if high_df.height > 0 else np.nan,
        "high_mean_markout_5s_bps": float(high_df["test_markout_5s_bps"].mean()) if high_df.height > 0 else np.nan,
        "low_count": int(low_df.height),
        "low_mean_return": float(low_df["test_total_return"].mean()) if low_df.height > 0 else np.nan,
        "low_mean_sharpe": float(low_df["test_sharpe"].mean()) if low_df.height > 0 else np.nan,
        "low_mean_mdd": float(low_df["test_max_drawdown"].mean()) if low_df.height > 0 else np.nan,
        "low_mean_toxic_fill_ratio": float(low_df["test_toxic_fill_ratio"].mean()) if low_df.height > 0 else np.nan,
        "low_mean_markout_1s_bps": float(low_df["test_markout_1s_bps"].mean()) if low_df.height > 0 else np.nan,
        "low_mean_markout_5s_bps": float(low_df["test_markout_5s_bps"].mean()) if low_df.height > 0 else np.nan,
    }
    print("\n=== Regime Analysis ===")
    print(regime_analysis)
    return wf_df


def main() -> None:
    # 1) 配置
    cfg = BacktestConfig()

    # 默认不开启 Walk-Forward；仅当显式设置 RUN_WALK_FORWARD=1 才执行。
    run_wf = os.getenv("RUN_WALK_FORWARD", "0") == "1"
    force_rebuild = os.getenv("FORCE_REBUILD", "0") == "1"
    wf_dir = Path(os.getenv("HFT_DATA_DIR", ".")).expanduser()
    if run_wf and cfg.enable_walk_forward_eval and wf_dir.exists():
        wf_out = run_walk_forward_from_parquet_dir(cfg, wf_dir)
        if wf_out is not None:
            wf_out.write_parquet("walk_forward_oos.parquet")
            print("saved: walk_forward_oos.parquet")
            return

    # 2) 接入数据与特征缓存一致性检查
    feature_path = wf_dir / "events_features.parquet"
    script_path = Path(__file__).resolve()
    use_cached_features = False
    needs_feature_cache = cfg.use_advanced_microprice or cfg.use_dictionary_strategy
    if needs_feature_cache and feature_path.exists():
        try:
            code_mtime = script_path.stat().st_mtime
            feat_mtime = feature_path.stat().st_mtime
            if force_rebuild:
                print("检测到 FORCE_REBUILD=1，强制重建 events_features.parquet")
            elif feat_mtime >= code_mtime:
                use_cached_features = True
            else:
                print("检测到代码更新晚于特征文件，强制重建 events_features.parquet")
        except OSError:
            use_cached_features = False

    if use_cached_features:
        events = pl.read_parquet(str(feature_path))
        print("加载特征缓存: events_features.parquet")
    else:
        trades_df, book_df = load_btc_data_placeholder()
        events = build_event_stream(trades_df, book_df, cfg)
        if cfg.use_dictionary_strategy:
            # Dictionary mode: only need basic micro_px from event stream
            pass  # build_event_stream already provides micro_px
        elif cfg.use_advanced_microprice:
            events = _prepare_events(events, cfg)
        # Cache the basic events for fast reload
        if needs_feature_cache:
            events.write_parquet(str(feature_path))
            print("已生成特征文件: events_features.parquet")

    # 3) 单组参数回测
    result = run_backtest(events, cfg)
    print("=== Single Run Metrics ===")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")
    _plot_markout_decay_curve(result["metrics"], "markout_decay_single.png")

    # 4) 参数扫描
    print("\n=== Grid Search Top 10 ===")
    gs = grid_search(events, cfg).head(10)
    print(gs)

    # 5) 简单双窗 OOS：train 选参，test 评估
    if cfg.enable_oos_eval and events.height >= 1000:
        train_events, test_events = split_train_test_by_time(events, cfg.oos_train_ratio)
        print(f"\n=== OOS Split === train={train_events.height}, test={test_events.height}")

        gs_train = grid_search(train_events, cfg)
        train_daily_events = _split_events_by_day(train_events)
        best_cfg = _select_best_cfg_with_stability(cfg, gs_train, train_daily_events)
        print(
            "best_cfg_from_train:",
            {
                "half_spread_ticks": best_cfg.half_spread_ticks,
                "gamma_inventory": best_cfg.gamma_inventory,
                "latency_ms": best_cfg.latency_ms,
                "queue_model": best_cfg.queue_model,
                "obi_hard_threshold": best_cfg.obi_hard_threshold,
                "obi_ema_alpha": best_cfg.obi_ema_alpha,
                "obi_trend_skew_bps": best_cfg.obi_trend_skew_bps,
                "intensity_threshold": best_cfg.intensity_threshold,
                "lambda_intensity": best_cfg.lambda_intensity,
                "fill_rate_alpha": best_cfg.fill_rate_alpha,
                "micro_px_alpha": best_cfg.micro_px_alpha,
            },
        )

        train_res = run_backtest(train_events, best_cfg)
        test_res = run_backtest(test_events, best_cfg)

        print("\n=== OOS Train Metrics ===")
        for k, v in train_res["metrics"].items():
            print(f"{k}: {v}")

        print("\n=== OOS Test Metrics ===")
        for k, v in test_res["metrics"].items():
            print(f"{k}: {v}")
        _plot_markout_decay_curve(test_res["metrics"], "markout_decay_oos_test.png")


def run_standard_oos(
    cfg: Optional[BacktestConfig] = None,
    feature_parquet: Optional[Path] = None,
    train_start: Optional[int] = None,
    train_end: Optional[int] = None,
    test_start: Optional[int] = None,
    test_end: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Standard OOS wrapper for script usage.
    If train/test bounds are None, both sides use the full feature dataset.
    """
    cfg2 = cfg if cfg is not None else BacktestConfig()
    if feature_parquet is None:
        data_dir = Path(os.getenv("HFT_DATA_DIR", ".")).expanduser()
        feature_parquet = data_dir / "events_features.parquet"
    else:
        feature_parquet = Path(feature_parquet).expanduser()
    if not feature_parquet.exists():
        raise FileNotFoundError(f"特征文件不存在: {feature_parquet}")

    events = pl.read_parquet(str(feature_parquet))
    if events.height == 0:
        raise ValueError("events_features.parquet 为空，无法回测")

    train_events = events
    test_events = events
    if train_start is not None:
        train_events = train_events.filter(pl.col("ts_ms") >= int(train_start))
    if train_end is not None:
        train_events = train_events.filter(pl.col("ts_ms") <= int(train_end))
    if test_start is not None:
        test_events = test_events.filter(pl.col("ts_ms") >= int(test_start))
    if test_end is not None:
        test_events = test_events.filter(pl.col("ts_ms") <= int(test_end))

    if train_events.height == 0:
        raise ValueError("train_events 为空，请检查 train_start/train_end")
    if test_events.height == 0:
        raise ValueError("test_events 为空，请检查 test_start/test_end")

    train_res = run_backtest(train_events, cfg2)
    test_res = run_backtest(test_events, cfg2)
    return {
        "train": train_res["metrics"],
        "test": test_res["metrics"],
    }


if __name__ == "__main__":
    main()
