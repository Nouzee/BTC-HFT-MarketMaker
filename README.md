# BTC-HFT-MarketMaker — Event-Driven Market Making Research Engine

基于逐 tick 事件驱动的做市策略回测与归因系统。  
覆盖撮合仿真、延迟建模、队列位置估计、参数网格搜索。

---

## 项目概述

在 1 个月 BTC 逐笔成交 + L2 盘口数据上，回测基于 micro-price 预测的被动做市策略。

核心工程挑战：
- 400 万 tick 级数据的事件驱动仿真
- 延迟与队列位置的真实建模
- 换手率约束下的参数优化

策略在 maker rebate (+0.5bps) 与结构性逆向选择 (~3.3bps) 的缺口下未能实现绝对收益，但建立了可复现的回测基础设施。

---

## 架构

```
Tick Data (trades + orderbook)
    │
    ▼
Event Replay Engine (Numba JIT)
    │
    ├── Latency Modeling
    ├── Queue Position Estimation
    └── Fill Simulation
    │
    ▼
Signal Layer
    │
    ├── Micro-price Prediction (quantile-based)
    └── Inventory Skew
    │
    ▼
Risk Controls
    │
    ├── OBI Circuit Breaker
    ├── Post-fill Cooldown
    └── Conservative Filling
    │
    ▼
Parameter Grid Search
    │
    ▼
Attribution & Reporting
```

---

## 核心实现

### 撮合引擎

- 事件驱动逐 tick 推进，非 bar 聚合
- 延迟建模：显式 `latency_ms` 模拟信号传播窗口
- 队列位置：`queue_ahead_ratio` + 时间衰减 → 排队成交概率
- Numba JIT 编译：400 万 tick 约 2 分钟完成单次回测

### Micro-price 预测

对 `imbalance` / `spread` / `total_depth` 做分位数分箱，离线学习条件期望：

```
G(I,S,V) = E[mid_{t+k} - mid_t | I, S, V]
```

- 对称技巧：买卖盘镜像翻转扩充训练样本
- 不对称报价：报价中心向 micro-price 偏移 + 库存 skew 回归中性

### 风控

- OBI 熔断：`|OBI| > 0.82` 时撤销对手方挂单
- 成交后冷却：普通 500ms，大单额外 400ms
- Conservative Filling：成交价格内化 0.8bps 预期逆向选择成本

---

## 实验结果

70% 训练 / 30% 测试，真实时间分割：

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| 月换手率 | 1287x | 1754x |
| 总收益 | -0.55% | -0.71% |
| Markout (100ms) | -3.48 bps | -3.20 bps |

测试集换手率略超 1500x 上限（测试集 tick 密度更高）。策略在 maker rebate 与逆向选择的结构性缺口下未能翻正，但风控逻辑将回撤控制在结构性下限附近。

---

## 项目结构

```
BTC-HFT-MarketMaker/
├── code/
│   ├── engine/          # 事件驱动撮合引擎 (Numba JIT)
│   └── scripts/         # 参数搜索 + 归因分析
├── data/                # 原始 tick 数据
├── results/             # 回测输出
├── requirements.txt
└── README.md
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 设置数据路径
export HFT_DATA_DIR="./data"
export HFT_TRADES_FILE="trades.csv"
export HFT_BOOK_FILE="book.csv"

python run_full_month.py
```

---

## 技术栈

Python · Numba · NumPy · Pandas  
Event-driven simulation · Queue position modeling · Grid search optimization

---

## 关于本项目

BTC-HFT-MarketMaker 是一个事件驱动做市策略研究引擎。项目的核心价值不在于策略的盈利结果（在当前市场结构下未能翻正），而在于建立了从逐 tick 仿真到参数优化的完整回测链路——包括延迟建模、队列位置估计和风控逻辑的工程实现。
