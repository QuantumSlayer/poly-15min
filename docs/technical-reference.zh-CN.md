# poly-15min：技术参考手册

[English version](technical-reference.md) · [项目 README](../README.zh-CN.md)

本手册说明已归档源码和模型文件的实际行为。目标读者应熟悉 Python、量化交易和 PyTorch。源码链接指向仓库中的对应文件。

> [!WARNING]
> **请勿将本系统连接到有资金的账户。** 系统会记录**明文 API 认证信息**，部分风控检查会直接终止进程，**却不撤销仍在盘口上的订单**。
>
> **使用本仓库的风险由使用者自行承担。** 本仓库仅供研究与算法交流，不提供任何保证或投资建议。**在适用法律允许的范围内，作者不对使用本仓库造成的交易损失或其他损害承担责任。** 具体条款以 [LICENSE](../LICENSE) 为准。

## 目录

| 章节 | 入口 |
|---|---|
| 01 | [项目状态与证据范围](#status) |
| 02 | [合约与定价假设](#contracts) |
| 03 | [源码结构与运行流程](#runtime) |
| 04 | [三个 Transformer 模型](#models)<br>[模型 A：收益率分布](#model-a) · [模型 B：短期分位数](#model-b) · [模型 C：盘口中价与半价差](#model-c) |
| 05 | [校准与报价构造](#quoting) |
| 06 | [订单执行与风险控制](#execution) |
| 07 | [配置与排错](#operations) |
| 08 | [训练与模型文件](#training) |
| 09 | [问题与安全风险](#issues) |

---

<a id="status"></a>
## 1. 项目状态、历史与证据

> [!IMPORTANT]
> **已归档，不再维护。** **本仓库中的全部代码和文档均由 AI 生成。** 代码通过**聊天界面使用 GPT-5 生成**；文档使用 **Opus 和 GPT-6** 生成。
>
> **代码与文档的正确性均未经人工审阅。** 作者从未审阅过用于实盘交易的代码。文档可能与实现存在冲突；了解系统的实际行为时，请以源码为准。
>
> 该策略曾在 BTC 上盈利，并于 **2025 年 12 月下旬**停止盈利。**其利润与头部账户相比，仅占极小的一部分。**
>
> 实盘系统曾持续修改。本仓库保留了大体相同的实现，但代码和模型参数可能与实际交易时使用的版本有所不同，不一定对应某个具体的实盘版本。

该策略曾在 BTC 上盈利，并于 2025 年 12 月下旬停止盈利。高频交易中，延迟至关重要：本系统使用交易所二手行情，比直接获取交易所数据的参与者至少慢 100 ms，这一差距抹去了原有的交易优势。

代码完全由 AI 生成，未经人工审阅。系统已归档，不再维护。

全部训练数据、模型 A 的训练入口脚本，以及模型 B/C 的数据集构建器已永久丢失。当前目录保留了推理和训练源码、六个 checkpoint、六个 JSON 元数据文件，但没有训练分片、历史成交记录或可复现的绩效实验。现存模型文件可用于结构检查。

<a id="evidence"></a>
### 证据与记号约定

系统行为以可执行源码为准，保存的架构与配置以 checkpoint 张量和 JSON 元数据为准。外部文档用于解释概念和兼容性约束。注释与代码冲突时，以代码为准。统计结果来自模型文件中保存的元数据。

所有特征下标均从零开始。张量形状中的 `N` 表示 batch size，`L` 表示序列长度，`D` 表示特征维度；合约时间中的 `B` 则表示下一个 15 分钟边界。Coinbase 标的价格以 USD 计，Binance 永续价格以 USDT 计。对数收益率、比值、概率和 Student-t 尺度均无量纲；`bp` 表示价格的万分之一。Polymarket 数量以 token 份额计。模型概率指导报价构造，可成交价格取决于订单簿。

围栏中的 `text` 内容用于展示数学伪代码或示意图。正弦位置编码在交替通道使用 `sin(pos / 10000^(2i/d_model))` 及对应余弦。输入和模型张量使用 float32，训练 AMP 选项可能改变计算精度；资产 ID 是 torch 整数索引，C 的 mask 为布尔值。

---

<a id="contracts"></a>
## 2. 合约、参考价格与定价假设

归档代码面向 BTC、ETH、SOL 和 XRP 的涨跌合约，每个窗口长 900 秒。`btc-updown-15m-1765306800` 这类 slug 包含资产、产品类别和窗口起点 `S`；到期时刻为 `B = S + 900`。代码使用 `ceil(now / 900) * 900`，所以时间戳恰好落在边界时，会返回该边界本身。实盘入口只允许交易 BTC，但行情采集和模型预测覆盖四种资产。[边界函数](../market_lib.py#L945)、[交易引擎初始化](../live_prediction.py#L216)。

目标合约的二元兑付结构是：获胜 token 每份兑付 1 美元，失败 token 兑付 0 美元。Up 表示结束参考价高于或等于开始参考价。代码并不实现交易场所的结算流程，也未保留所有历史合约规则；具体合约仍应以其自身规则为准。YES/NO 的兑付互补。两个订单簿独立形成的买价、卖价、最新成交价或实际成交价格之和可能高于或低于 1 美元。买入 NO 与卖出 YES 的方向暴露相关，但所需现金、抵押品和 token 持仓并不相同。

<a id="reference-prices"></a>
### 参考价格与时间

Coinbase 的 **microprice** 是用对侧挂单量为 best bid/ask 加权得到的价格估计：`(ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)`。ticker 处理器依次回退到算术中价、成交价。另一方面，L2 特征处理器在双边盘口有效时，用算术中价作为深度区间的中心；该算术中价保存在局部变量 `micro` 中。[Ticker 处理](../market_lib.py#L1820)、[L2 特征](../market_lib.py#L1426)。

| 数值 | 计算方式 | 含义 |
|---|---|---|
| `coinbase` | hub 中保存的最新 ticker microprice | 当前标的价格的近似值 |
| `prev_close` | 本地观测中，时间不晚于 `S - 1.1` 秒的最后一个价格 | 本地采样得到的涨跌参考价；交易场所采用其指定的预言机参考价 |
| `prev_low`、`prev_high` | `[S - 2, S - 1]` 内本地观测的最低/最高价；缺失时回退到 close | 短采样区间内的参考价格范围 |
| `tau` | 交易快照使用本地发送时间，校准器使用 Coinbase 服务器时间；均减 1.1 秒，并将定价所用的剩余时间设为至少一秒 | 基于不同时钟计算的两套剩余期限近似值 |
| `chainlink_15m_close` | 在边界附近选取的 RTDS 观测 | 用于边界分析的预言机参考价记录 |

本地 Coinbase 历史通过轮询任务采样价格变化，启动时缺少历史回填。因此，在窗口中途启动时，可能要等到后续边界才能获得 `prev_close`。代码采用了 1.1 秒偏移，但现有数据不足以确认其实际效果。Binance 永续、Coinbase 与 Chainlink 之间的差异会引入参考价格和时间对齐风险。[参考价提取](../live_prediction.py#L344)、[快照时间](../live_prediction.py#L1181)。

交易 worker 从带本地时间戳的队列取得 `ts_s`，校准器则从 `coinbase_ts_server_ms` 取得时间。虽然注释声称使用相同期限，延迟的交易所时间戳仍可能产生不同 `tau`，甚至指向不同窗口边界。模型 A 则使用已完成聚合的 Binance 秒级时间戳构造期限输入。

Chainlink 服务优先选择边界之前或恰好位于边界的最后一个 tick，也允许回退到边界后五秒内的 tick。但调度器只等待约 0.25 秒，因此并不保证会收集完整的五秒后续数据。该服务负责写入参考价记录和 hub 状态；token 结算由交易场所负责。其日志调用重复传入 `event` 参数，可能在文件和状态已经写入后抛出异常。[服务实现](../market_lib.py#L966)、[记录路径](../market_lib.py#L1025)。

<a id="probability"></a>
### Student-t 概率与定价假设

模型 A 用位置参数为零的 Student-t 分布描述对数收益率。对于正的标的价格 `P`、涨跌参考价 `K`、剩余时间 `tau`、尺度 `iv` 和自由度 `nu`，定价规则如下。`p_up(P, K)` 显式写出标的价格和涨跌参考价；改变任一价格时，`iv`、`nu` 和 `tau` 保持不变。

```text
s_tau = iv * sqrt(max(tau, 1) / 900)
r_star = log(K / P)
p_up(P, K) = 1 - F_nu(r_star / s_tau)
```

`iv` 表示 900 秒对数收益率的尺度参数。只有 `nu > 2` 时，标准差才是 `s_tau * sqrt(nu / (nu - 2))`；只有 `nu > 1` 时，均值才存在。均值存在时，该分布的均值为零。模型的输出变换并未强制满足这两个自由度下限。[模型](../utils.py#L888)、[概率函数](../market_lib.py#L46)、[SciPy 的 Student-t 参数定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)。

由对称性可知，`P = K` 时 `p_up = 0.5`，`P > K` 时 `p_up > 0.5`。时间平方根缩放与零位置参数都是建模选择，其对这些合约的适用性仍待实证检验。基于历史收益率分布进行风险中性估值还需要额外假设。这条规则未纳入手续费、流动性、排队位置、抵押品成本和结算不确定性；输出用于指导策略，实际交易价值还取决于这些未纳入的因素。本系统根据收益率分布计算二元事件概率。

---

<a id="runtime"></a>
## 3. 源码结构、运行任务与数据

| 源文件 | 行数 | 职责 |
|---|---|---|
| [live_prediction.py](../live_prediction.py) | 1,607 | 启动、模型调度、快照组装、向交易引擎传递数据 |
| [live_lib.py](../live_lib.py) | 1,077 | 模型 A 加载、Binance 特征、HTTP、日志 |
| [market_lib.py](../market_lib.py) | 2,551 | 共享状态、行情源、市场发现、校准 |
| [utils.py](../utils.py) | 1,356 | 模型 A、历史特征、训练、Student-t 数值计算 |
| [train_seq.py](../train_seq.py) | 1,131 | 模型 B 训练与指标 |
| [quote_seq.py](../quote_seq.py) | 872 | 模型 B 推理与方向策略 |
| [train_iv.py](../train_iv.py) | 1,225 | 模型 C 训练与指标 |
| [quote_iv.py](../quote_iv.py) | 1,013 | 模型 C 事件重放与推理 |
| [trade.py](../trade.py) | 1,819 | 报价构造、订单调度、库存过滤 |
| [trade_lib.py](../trade_lib.py) | 1,448 | API 认证、CLOB I/O、成交追踪、风控线程 |
| [hub_debug.py](../hub_debug.py) | 135 | 定期记录 hub 的公开属性快照 |

<a id="architecture"></a>
### 系统架构与启动

```text
Binance 永续成交 + 分钟 K 线 --> 模型 A --> 原始 iv、df --> Student-t 定价
                                      \--> PM 校准 --> 乘子 --> 库存限制偏移
                                                   \--> 修正尺度 --> 模型 C
Coinbase ticker + L2 + 成交 --> 模型 B --> 方向偏移 --------------------------\
Coinbase 特征 + PM 盘口 + 模型 A iv --> 模型 C --> 中价 + 半价差 --------------> 报价
PM 盘口 ----------------------------------> 校准与报价检查 ----------------/
报价 --> TradeEngine --> CLOB GTC 提交 / 撤单请求
用户 WebSocket 成交 --> 本地库存估计 --> 交易动作过滤
持仓 API --> 独立的强制退出检查
Chainlink RTDS --> 边界参考价记录
```

模型 C 的特征定义还包含 `model_df`，但实时更新路径没有写入它。图中展示实际生效的输入路径。执行引擎使用模型 A 的原始尺度计算 Student-t 概率。校准乘子用于调整库存限制，修正尺度则作为模型 C 的输入。

启动时先创建日志和 hub，启动调试与校准任务，初始化仅交易 BTC 的交易引擎，安装预测日志转接函数，并启动参考价与 Coinbase 服务。之后初始化模型 B/C、tick 生产者、各资产 worker、市场发现与 CLOB 订阅。四个模型 A checkpoint 及各资产 120 分钟 REST 回填依次处理，随后启动错峰的分钟更新器，预热五秒后再启动 Binance 成交流。秒级环形缓冲仍需等待实时观测填满。[入口](../live_prediction.py#L190)、[模型 A 启动](../live_prediction.py#L1496)。

顶层共创建 22 个协程实例：hub 快照、校准器、两个 Chainlink 任务、Coinbase 记录器、模型 C 清理器、tick 生产者、heartbeat、四个做市 worker、slug 刷新器、CLOB 流、四个分钟更新器和四个 Binance 流。L2 处理器、prediction drain task、执行任务、风控循环和 I/O 线程还会创建额外实例。模型 A 优先选择 CUDA，否则使用 CPU；实盘模型 B/C 显式使用 CPU。PyTorch 算子内与算子间线程数均设为三，但这并不能保证事件循环及时响应。

<a id="shared-state"></a>
### 共享状态与调度

`StateHub` 使用可变字典，设计上要求由主 asyncio 循环更新。每个资产的队列容量为一，中间快照会被丢弃。CLOB 同步回调向模型 C 写入事件；合并预测请求的 prediction drain task 发布 `seq_quote` 和 `iv_pred`。这些预测结果保存在运行时附加到 hub 的属性中。模型推理在事件循环线程上同步执行，即使所在函数使用没有暂停点的 `async def` 也是如此。模型 C 的 `RLock` 覆盖事件插入和输入组装；前向计算在该锁之外执行。[Hub](../market_lib.py#L191)、[C 推理](../quote_iv.py#L969)。

模型 A 通过 monkey patch 发布结果：替换后的 `live_mod.jlog` 先记录 `prediction`，再把 `iv`、`df` 和交易所时间写入 hub。此前通过 `from live_lib import jlog` 导入的引用仍指向原函数。经过转接函数的断线事件可清空预测；独立的新鲜度检查也会移除超过 20 秒的预测。[转接函数](../live_prediction.py#L251)。

tick 生产者每 2 ms 轮询一次。每次观测到的价格变化都会进入本地历史缓冲，但只有达到 `MP_THRESH` 的变化才进入下游模型和交易路径：BTC 0.1、ETH 0.01、SOL 0.01、XRP 0.0001。heartbeat 每 50 ms 检查一次，在连续 0.32 秒没有发送后补发。实际调度延迟可能超过这些目标值。heartbeat 会复用已有价格，所以即使交易所行情没有更新，模型窗口也可能继续获得输入。快照中的 `cb_last_ts` 是本地发送时间。[生产者](../live_prediction.py#L1070)。

做市 worker 要求模型 A 的预测不超过 20 秒、Polymarket 数据不超过 10 秒，却没有对 Coinbase 设置同等独立的时效性检查。模型 C 的缓存预测超过约 700 ms 时，会尝试同步补算；补算失败后，旧缓存仍可能继续使用。模型 B/C 的默认最小预测间隔均为每资产 50 ms。系统没有用于区分新旧合约的统一标识，无法阻止已经运行的 prediction drain task 在缓存清空后写回旧结果。

<a id="feeds"></a>
### 行情源、订单簿与合约切换

| 数据源 | 归档代码中的地址/频道 | 处理与失败行为 |
|---|---|---|
| Binance USD-M 永续 | `fstream.binance.com`、`aggTrade`；`fapi.binance.com` 永续/指数 K 线 | 按成交时间构建秒数据；异常重连触发 reseed；REST 异常最多尝试三次、超时 10 秒，非 200 响应直接返回 |
| Coinbase Advanced Trade | `advanced-trade-ws.coinbase.com`；`heartbeats`、`ticker`、`level2`、`market_trades` | ticker microprice 与收益率；独立 L2 订单簿；固定一秒间隔重连 |
| Polymarket CLOB | `ws-subscriptions-clob.polymarket.com/ws/market` | 全量快照与档位绝对值更新，保留完整深度；价格键保留六位小数 |
| Gamma 市场发现 | `gamma-api.polymarket.com/public-search`、`/markets/slug/{slug}` | 搜索/探测结果写入 `temp/`；请求超时 15 秒 |
| Chainlink 经 RTDS | `ws-live-data.polymarket.com`、`crypto_prices_chainlink` | 兼容多种消息结构，记录边界参考价 |

Coinbase L2 更新进入无界队列。每条更新都会修改本地盘口，但派生特征按交易所时间最多每 0.10 秒重算一次。代码携带序列号，却不检查序列缺口。snapshot 会重置订单簿；重连与缺口恢复机制仍不完整。成交窗口只在新成交到达时清理过期记录，所以标为“最近 1/3/5 秒”的统计量可能在两次成交之间保持旧值。代码直接使用 feed 提供的 BUY/SELL 字段；本仓库无法确认该历史接口对这一字段的买卖方向定义。[Coinbase 记录器](../market_lib.py#L1257)。

Gamma 发现流程会在需要时探测相邻窗口。其等待时间算法可能使刷新间隔达到约 15–30 分钟。CLOB 本地市场读取器在选择 YES token 之前丢弃了 outcome 元数据，因而实际会回退到第一个 token；执行端独立的 `TokenIndex` 则会解析 Up/Down outcome。如果 token 顺序不符合假设，两条路径可能选择不同 token。[发现流程](../market_lib.py#L1905)、[本地读取器](../market_lib.py#L1975)、[token 索引](../trade_lib.py#L144)。

在边界 `B` 附近，旧 CLOB reader 约于 `B - 10` 停止，起点为 `B` 的新窗口约于 `B - 5` 开始订阅。模型 C 在 `B + 1` 清空，模型 B 保留。新 snapshot 到达前，已有 PM 盘口不会被清空，也没有标上新 slug。回调只在 best bid/ask 价格变化时触发，所以仅数量变化会更新 hub，却不会直接产生模型 C 的 book 事件。WebSocket 陈旧检查采用所有资产中最新的时间戳；交易 worker 则另有逐资产检查。[CLOB 流](../market_lib.py#L1998)、[C 清理](../live_prediction.py#L1011)。

---

<a id="models"></a>
## 4. 三个 Transformer 模型

| 属性 | 模型 A | 模型 B | 模型 C |
|---|---|---|---|
| 序列 | `(N,300,14)` | `(N,32,75)` | `(N,64,22)` |
| 其他输入 | `(N,19)` 静态向量 | `(N,229)` 静态向量；`(N,)` 资产 ID | `(N,64)` 有效位置 mask；`(N,)` 资产 ID |
| 宽度 / attention heads / 层数 / FFN | 96 / 4 / 4 / 288 | 256 / 8 / 4 / 512 | 192 / 8 / 4 / 384 |
| 归一化 / readout | post-norm / attention pooling | pre-norm / last token | pre-norm / last token |
| 输出 | `iv`、`df` | 3 个期限 × 5 个分位数估计 | 中价残差、半价差 |
| 可训练参数 | 每资产 420,995 × 4 | 2,596,156 | 1,492,808 |
| 资产 ID | 每个资产独立模型 | BTC 0、ETH 1、XRP 2、SOL 3 | BTC 0、ETH 1、SOL 2、XRP 3 |

三个模型都使用不带因果 mask 的编码器 self-attention，每个位置都可以关注输入历史窗口中的全部位置。全部输入观测都应在预测时刻已经可用。只有模型 C 提供 padding mask。编码器 FFN 使用 ReLU，外部 MLP 则采用不同激活函数。标准 PyTorch 编码器层包含 attention/output 投影、两个残差连接、两个 LayerNorm、FFN 偏置和 dropout。每个 attention head 的维度分别为 A 24、B 32、C 24。padding mask 和优化 kernel 的行为可能随 PyTorch 版本变化，而仓库没有固定版本。

---

<a id="model-a"></a>
### 模型 A：分布参数

`SeqTransformerT` 预测位置参数为零的 Student-t 收益率分布的尺度与自由度。四个独立 checkpoint 的架构相同，但权重、scaler 和资产代码不同。历史输入来自 Binance 永续合约，尽管部分辅助函数名含有 `spot`；输入中没有资金费率特征。[架构](../utils.py#L888)、[特征构建](../utils.py#L267)、[秒级聚合](../utils.py#L352)。

<a id="a-inputs"></a>
#### A 输入：特征顺序与定义

序列顺序由 `seconds_feature_cols()` 与 `SEC_COLS` 固定。价格单位是每单位标的的 USDT，数量单位是标的币。历史聚合会重建完整的一秒网格，价格前向填充，无成交活动填零。收益率是对数收盘价之差；`sec_trades` 统计 aggregate trade 记录数，每条记录可能合并多笔撮合。[列顺序](../utils.py#L1185)、[实时列定义](../live_lib.py#L45)。

| 下标 | 特征 | 原始含义与单位 |
|---|---|---|
| 0 | `sec_close` | 该秒最后成交价；USDT |
| 1 | `sec_vwap` | 价格 × 数量之和除以总数量；USDT |
| 2 | `sec_vol` | 总成交数量 |
| 3 | `sec_signed_vol` | 买入数量减卖出数量 |
| 4 | `sec_trades` | aggregate trade 记录数 |
| 5 | `sec_buy_vol` | 买方未标为 maker 的成交数量 |
| 6 | `sec_sell_vol` | 买方标为 maker 的成交数量 |
| 7 | `sec_ret1` | 一步对数收益率 |
| 8 | `sec_ret3` | 三步对数收益率 |
| 9 | `sec_ret5` | 五步对数收益率 |
| 10 | `sec_ret10` | 十步对数收益率 |
| 11 | `sec_ret15` | 十五步对数收益率 |
| 12 | `sec_ret30` | 三十步对数收益率 |
| 13 | `sec_imb` | `(buy - sell) / (buy + sell + 1e-9)`；已成交买入量与卖出量的失衡 |

静态输入由 14 个分钟特征和五个剩余期限特征组成。分钟滚动标准差使用 pandas 的样本标准差约定，没有年化处理。[分钟特征](../utils.py#L267)、[期限特征](../utils.py#L1238)。

离线分钟行只要缺少必需值就会被删除，包括滚动窗口预热行和未匹配到指数价格的行。振幅计算将零收盘价视为缺失。60 分钟对数成交量 z-score 的分母是样本标准差加 `1e-9`；时间角度为 `2*pi*UTC_minute_of_day/1440`。

| 下标 | 特征 | 原始含义与单位 |
|---|---|---|
| 0 | `ret_1` | 一分钟对数收盘价差 |
| 1 | `ret_3` | 三分钟对数收盘价差 |
| 2 | `ret_5` | 五分钟对数收盘价差 |
| 3 | `rv_5` | 最近五行一分钟收益率的滚动标准差 |
| 4 | `rv_15` | 最近十五行的同类标准差 |
| 5 | `rv_ratio` | `rv_5 / (rv_15 + 1e-9)` |
| 6 | `hl_range` | `(high - low) / close` |
| 7 | `taker_buy_ratio` | taker 买入量 / `(volume + 1e-9)`，截断到 `[0,1]` |
| 8 | `trades` | 分钟 K 线成交笔数 |
| 9 | `vol_z_60` | `log(volume + 1e-12)` 的 60 分钟滚动 z-score |
| 10 | `basis_rel` | `(futures_close - index_close) / (index_close + 1e-9)` |
| 11 | `tod_sin` | UTC 日内分钟角度的正弦 |
| 12 | `tod_cos` | UTC 日内分钟角度的余弦 |
| 13 | `is_weekend` | 周六/周日标记 |
| 14 | `tau` | 剩余秒数 |
| 15 | 期限变换 | `sqrt(tau / 900)` |
| 16 | 期限变换 | `sqrt(900 / max(tau,1))` |
| 17 | 期限变换 | `sin(2*pi*tau/900)` |
| 18 | 期限变换 | `cos(2*pi*tau/900)` |

训练集上的两组 `StandardScaler` 统计量分别覆盖按样本/时间展平的序列通道，以及静态向量。模型文件保存长度为 14 和 19 的均值/尺度数组，state dictionary 中另有共 66 个 float32 scaler buffer 元素。实时 `NumpyScaler` 使用 `(X - mean) / (scale + 1e-9)`；原始价格水平直接参与这一标准化计算。如果 scaler 重建失败，辅助函数可能回退到原始输入，因此模型成功构建并不能证明预处理正确。分钟行缺失时，除特定的上一分钟宽限规则外，不执行推理。[Scaler](../utils.py#L1260)、[实时加载](../live_lib.py#L290)、[输入变换](../live_lib.py#L817)。

<a id="a-network"></a>
#### A 网络层与输出

```text
(N,300,14) --> Linear 14->96 --> 加性正弦 PE --> 4 层 post-norm 编码器
             --> attention pooling --> (N,96)
(N,19) --> Linear 19->64 --> SiLU --> ResidualMLP --> Linear 64->32 --> SiLU --> (N,32)
concat --> (N,128) --> Linear 128->64 --> SiLU --> ResidualMLP --> Dropout .2 --> Linear 64->2
raw[:,0] --> .0005 * softplus --> iv
raw[:,1] --> 16 / (softplus + 1e-8) --> df
```

每个 `ResidualMLP` 为 `LayerNorm(x + Linear96to64(Dropout(SiLU(Linear64to96(x)))))`，dropout 为 0.2。attention pooling 沿时间维计算 `softmax(v(tanh(W h)))`，再对编码器表示做加权求和；`W` 为 96→96，`v` 为 96→1。保存到 checkpoint 的位置编码 buffer 形状为 `(1,4096,96)`。`forward` 中没有实际应用 `IV_FLOOR`；docstring 中的 0.0004 系数有误，可执行常量为 0.0005。[网络层定义](../utils.py#L849)。

四个 A checkpoint 内记录的配置都是 `sec_d=14`、`static_d=19`、`d_model=96`、`n_head=4`、`depth=4`、`dropout=0.2`、`l_sec=300`。构造函数默认值则是宽度 64、四个 attention heads、三层编码器，FFN 宽度为 `3*d_model`。checkpoint 记录了对构造函数默认值的覆盖；训练入口丢失后，当时传入这些参数的具体方式已无法还原。

| 组件 | 元素数 |
|---|---|
| 输入投影 | 1,440 |
| 四层编码器 | 373,248 |
| Attention pooling | 9,409 |
| 静态 MLP | 15,936 |
| 融合 MLP | 20,832 |
| 输出层 | 130 |
| 可训练参数合计 | 420,995 |
| 位置编码 buffer | 393,216 |
| Scaler buffers | 66 |
| 保存的 state 合计 | 814,277 |

<a id="a-training"></a>
#### A 采样、损失与训练

对每个候选分钟索引，`make_samples` 从 5 到 900 均匀抽取整数 `tau`。保存的元数据规定每个分钟索引抽取 12 次，库默认值为一次。函数设 `end = floor_15min(minute) + 900 seconds - 1 second`，再设 `t = end - tau`。因此，样本可能落在候选分钟之外，不同抽样也可能得到同一时间戳。标签为 `r = log_close[end] - log_close[t]`、`z_true = r / (sqrt(tau/900) + 1e-12)`。[采样](../utils.py#L1195)、[数据集](../utils.py#L1213)。

现有离线分钟特征构建器使用完整分钟 OHLC，却按分钟开始时间建索引；`indexify_samples` 又选择 `t.floor('min')`。将这些辅助函数直接用于分钟内样本时，输入会包含该分钟内晚于样本时刻的信息。这是现有辅助函数组合中已经确认的前视数据路径；由于训练入口丢失，无法确定历史 checkpoint 数据集的具体构建方式。日期划分会随机选取 20% 的 UTC 日期作为验证集，并在两侧各隔离 900 秒，但这种隔离并不能解决分钟内前视问题。[索引](../utils.py#L157)、[划分](../utils.py#L554)。

损失先把按期限归一化的标签还原为原始期限收益率，再计算加权 Student-t 负对数似然：

```text
r_true = z_true * sqrt(max(tau,1)/900)
s = iv * sqrt(max(tau,1)/900)
u = r_true / (s + 1e-12)
log_f = lgamma((nu+1)/2) - lgamma(nu/2) - .5*log(nu*pi) - log(s+1e-12)
        - .5*(nu+1)*log1p(u*u / max(nu,1e-8))
w0 = clip(sum_k sqrt(abs(log_close[t]-log_close[t-k])/k + 1e-15), 1e-6, 50)
w = clip(w0 * prob_mult, 1e-8, 1e3)
loss = sum(w * -log_f) / (sum(w) + 1e-12)
k in {1,3,5,10,15,30}
```

默认 `r_thresh_bp=0` 时，概率加权更新会把初始值为一的乘子统一改为 0.05，因此没有引入随预测变化的权重。公共系数在理想的归一化均值中会抵消，但截断和分母 epsilon 意味着不能保证严格抵消。验证使用未加权 NLL，与训练目标不同。[训练循环](../utils.py#L982)。

库默认使用 AdamW，`lr=1e-3`、`wd=5e-4`，训练四个 epoch，早停 patience 为三，改进阈值为 `1e-4`。可选余弦调度逐 batch 更新，`T_max = epochs * len(train_loader)`，最低学习率为初始值的 0.1 倍。该函数没有 AMP 或梯度裁剪。它把最佳状态复制到 CPU，并在结束时重新加载。采样和划分默认 seed 为 42，但工具函数不设置全局 torch seed。模型文件保留了八个 epoch 摘要，却无法证明丢失的训练入口是否覆盖了优化器配置。辅助函数在默认零 worker 的 DataLoader 上无条件设置 `prefetch_factor=2`，存在依赖版本相关的失败路径。

训练函数接收外部已构建的 loader，因此自身不设置训练 batch size。`fit_scalers_lazy` 默认 batch 为 512，并打乱样本；`predict_params_t_dataset` 默认 batch 为 4096，不打乱样本。两者默认零 worker，保留最后一个不足整批的 batch，在 CUDA 可用时启用 pinned memory，仅在 worker 数为正时启用 persistent workers。概率权重默认每个 epoch 更新一次，另外支持 `batch` 和 `off` 模式。历史训练入口所用的 loader 配置已不可得。

<a id="a-runtime"></a>
#### A 实时行为与校准诊断

加载器先按路径选择匹配资产的 checkpoint，找不到时可能回退到任意资产中最新的 checkpoint。它剥离 `_orig_mod.base.` 或 `base.` 前缀，按相应分支去掉 wrapper 状态，再严格加载基础网络。这个回退可能在不报错的情况下选错资产。当前模型文件的结构检查不需要执行序列化的 scaler 对象。[加载器](../live_lib.py#L238)。

实时环形缓冲在不同秒的成交到达时完成上一秒的聚合，先计算收益率，再追加当前收盘价；积累 300 行后才预测。它不补齐空秒，也不拒绝倒退时间戳，因此实时序列中相隔若干行，不一定对应历史序列中相同的秒数间隔。异常重连会重置缓冲并重新回填 120 分钟的分钟数据；正常 WebSocket 关闭不走完全相同的 reseed 路径。[缓冲](../live_lib.py#L619)、[成交流](../live_lib.py#L949)。

分钟更新器在边界后等待两秒，并对缺失数据重试，但它直接取 REST 返回的最后一根 K 线，不检查该 K 线是否已经收盘。它只接收开盘时间更新的行，因此尚未收盘的 K 线可能被固定为该分钟的特征。静态输入优先取当前分钟行，或在边界后十秒内取紧邻的上一分钟行。嵌套 HTTP 超时可能使总等待时间超过七秒。[更新器](../live_lib.py#L709)、[静态行选择](../live_lib.py#L855)。

PIT 为 `F_nu(r/s)`。名为 `ece_uniform` 的函数计算每个非空分箱中 PIT 均值与分箱中点的距离，再按该箱样本占比加权。`mce` 是最大的这类箱内距离。因此，即使所有观测都集中在一个分箱中点，误差也可以为零，而分布显然不均匀。KS 比较经验 CDF 与均匀分布 CDF；重叠窗口和序列相关会削弱其 p 值通常所依赖的独立样本解释。较小的历史分数不能证明校准良好或具备交易价值。[指标实现](../utils.py#L757)。

每个 A checkpoint 还保存了一个在 0.5 处分段的三次 PIT 映射，并对 0、0.5、1 施加端点约束。拟合入口已经丢失。实时加载不应用该映射；它与[第五章](#calibration)的市场价格校准是两个不同机制。

---

<a id="model-b"></a>
### 模型 B：短期分位数估计

模型 B 按 `ret_2s`、`ret_5s`、`ret_10s` 的顺序，对每个目标输出五个估计值，意图对应未来对数收益率的 `[0.10,0.25,0.50,0.75,0.90]` 分位数。数据集构建器丢失，因而无法核实标签的精确时间对齐和最初采用的目标价格定义。实时预处理可从源码查阅；它与历史训练流程的一致性仍待核实。[训练模型](../train_seq.py#L397)、[推理](../quote_seq.py#L234)。

<a id="b-inputs"></a>
#### B 输入：特征顺序与定义

下表 75 列顺序取自 [meta_transformer.json](../data_seq32/meta_transformer.json)，生产端为 [Coinbase 处理流程](../market_lib.py#L1257)，接入端为 [add_tick](../quote_seq.py#L420)。`Q` 表示标的币数量，`R` 表示无量纲比值或对数收益率，`bp` 表示基点；计数和标记无量纲。所有通道随后都按该资产保存的统计量标准化。

当 L2 中价 `M` 有效时，深度区间包含价格不低于 `M*(1-bp/10000)` 的买单，以及不高于 `M*(1+bp/10000)` 的卖单。`tot = bid + ask`，`imb = (bid-ask)/tot`（分母为零时取零），`lr = log(bid+1e-9)-log(ask+1e-9)`。`mp_skew` 字段使用同样的区间深度失衡算法。中价缺失或无效时，代码可能改用单侧价格或最近的 ticker microprice。

| 下标 | 特征 | 含义；原始单位 |
|---|---|---|
| 0 | `cb_book_avg_dist_ask_bp` | 前五档卖单按数量加权的价格距离；bp |
| 1 | `cb_book_avg_dist_bid_bp` | 前五档买单按数量加权的价格距离；bp |
| 2 | `cb_book_convexity` | 买卖两侧挂单量对价格距离的斜率之和；Q/bp |
| 3 | `cb_book_slope_ask` | 前五档卖单数量对距离的 OLS 斜率；Q/bp |
| 4 | `cb_book_slope_bid` | 买侧对应斜率；Q/bp |
| 5 | `cb_buy_frac_1s` | 最近成交窗口中 BUY 数量占比；R |
| 6 | `cb_buy_frac_3s` | 三秒窗口的对应占比；R |
| 7 | `cb_buy_frac_5s` | 五秒窗口的对应占比；R |
| 8 | `cb_depth_ask_10bp` | 10 bp 内卖侧深度；Q |
| 9 | `cb_depth_ask_1bp` | 1 bp 内卖侧深度；Q |
| 10 | `cb_depth_ask_2bp` | 2 bp 内卖侧深度；Q |
| 11 | `cb_depth_ask_5bp` | 5 bp 内卖侧深度；Q |
| 12 | `cb_depth_bid_10bp` | 10 bp 内买侧深度；Q |
| 13 | `cb_depth_bid_1bp` | 1 bp 内买侧深度；Q |
| 14 | `cb_depth_bid_2bp` | 2 bp 内买侧深度；Q |
| 15 | `cb_depth_bid_5bp` | 5 bp 内买侧深度；Q |
| 16 | `cb_depth_imb_10bp` | 10 bp 内深度失衡；R |
| 17 | `cb_depth_imb_1bp` | 1 bp 内深度失衡；R |
| 18 | `cb_depth_imb_1bp_diff` | 相对上次特征计算的变化；R |
| 19 | `cb_depth_imb_2bp` | 2 bp 内深度失衡；R |
| 20 | `cb_depth_imb_5bp` | 5 bp 内深度失衡；R |
| 21 | `cb_depth_lr_10bp` | 10 bp 内买卖深度的对数比；R |
| 22 | `cb_depth_lr_1bp` | 1 bp 内对应对数比；R |
| 23 | `cb_depth_lr_2bp` | 2 bp 内对应对数比；R |
| 24 | `cb_depth_lr_5bp` | 5 bp 内对应对数比；R |
| 25 | `cb_depth_near_far_ratio_ask` | 卖侧 1 bp 深度 / 5 bp 深度；R |
| 26 | `cb_depth_near_far_ratio_bid` | 买侧 1 bp 深度 / 5 bp 深度；R |
| 27 | `cb_depth_tot_10bp` | 10 bp 内买卖深度之和；Q |
| 28 | `cb_depth_tot_1bp` | 1 bp 内对应总深度；Q |
| 29 | `cb_depth_tot_2bp` | 2 bp 内对应总深度；Q |
| 30 | `cb_depth_tot_5bp` | 5 bp 内对应总深度；Q |
| 31 | `cb_flow1s_net` | 一秒内 BUY 数量减 SELL 数量；Q |
| 32 | `cb_flow3s_net` | 三秒内对应数量差；Q |
| 33 | `cb_flow5s_net` | 五秒内对应数量差；Q |
| 34 | `cb_jump_flag` | ticker 收益率绝对值超过更新后 EWMA sigma 的三倍；0/1 |
| 35 | `cb_last_trade_at_ask` | 最新成交价与当前卖一价相差不超过 `1e-8`；0/1 |
| 36 | `cb_last_trade_at_bid` | 最新成交价与当前买一价相差不超过 `1e-8`；0/1 |
| 37 | `cb_last_trade_px` | 最新成交价除以当前 ticker microprice；R |
| 38 | `cb_last_trade_side` | BUY 为 +1，SELL 为 -1，其他为 0 |
| 39 | `cb_last_trade_ts_s` | 最新成交时间；重设原点后的秒数 |
| 40 | `cb_last_trade_vs_mid_bp` | `(last_trade_px-ticker_microprice)/ticker_microprice*10000`；bp |
| 41 | `cb_last_ts` | 本地下游发送时间；重设原点后的秒数 |
| 42 | `cb_mp_skew_10bp` | 10 bp 内区间深度失衡；R |
| 43 | `cb_mp_skew_1bp` | 1 bp 内对应失衡；R |
| 44 | `cb_mp_skew_2bp` | 2 bp 内对应失衡；R |
| 45 | `cb_mp_skew_5bp` | 5 bp 内对应失衡；R |
| 46 | `cb_n_ask_improve_1s` | 一秒内各次特征更新中卖一价下降次数 |
| 47 | `cb_n_ask_worsen_1s` | 一秒内各次特征更新中卖一价上升次数 |
| 48 | `cb_n_bid_improve_1s` | 一秒内各次特征更新中买一价上升次数 |
| 49 | `cb_n_bid_worsen_1s` | 一秒内各次特征更新中买一价下降次数 |
| 50 | `cb_n_spread_tighten_1s` | 一秒内正价差缩小次数 |
| 51 | `cb_n_spread_widen_1s` | 一秒内正价差扩大次数 |
| 52 | `cb_net_add_ask_1bp_1s` | 一秒内卖侧区间深度变化之和；Q |
| 53 | `cb_net_add_bid_1bp_1s` | 买侧对应深度变化之和；Q |
| 54 | `cb_ret_10s` | 当前 ticker 对数价格减去保留历史中 `t-10` 及之后的第一个对数价格；R |
| 55 | `cb_ret_1s` | 使用 `t-1` 的对应收益率；R |
| 56 | `cb_ret_3s` | 使用 `t-3` 的对应收益率；R |
| 57 | `cb_ret_5s` | 使用 `t-5` 的对应收益率；R |
| 58 | `cb_rv_3s` | 三秒内事件收益率平方和的平方根；R |
| 59 | `cb_rv_dn_3s` | 仅计负收益率的对应数值；R |
| 60 | `cb_rv_up_3s` | 仅计正收益率的对应数值；R |
| 61 | `cb_sigma_ewma` | 事件收益率的 EWMA 离散程度；R |
| 62 | `cb_spread_abs` | L2 卖一价减买一价，再除以 ticker microprice；R |
| 63 | `cb_spread_bp` | L2 价差 / L2 中价 × 10000；bp |
| 64 | `cb_tob_ask_px` | 卖一价除以 ticker microprice；R |
| 65 | `cb_tob_ask_qty` | 卖一档数量；Q |
| 66 | `cb_tob_bid_px` | 买一价除以 ticker microprice；R |
| 67 | `cb_tob_bid_qty` | 买一档数量；Q |
| 68 | `cb_ts_server_ms` | 服务器毫秒时间转换为重设原点后的秒数 |
| 69 | `cb_wall_ask_dist_bp` | 25 bp 内数量最大的卖侧聚合档位距中价的距离；bp |
| 70 | `cb_wall_ask_size` | 该价格档位聚合的总数量；Q |
| 71 | `cb_wall_bid_dist_bp` | 25 bp 内数量最大的买侧聚合档位距中价的距离；bp |
| 72 | `cb_wall_bid_size` | 该价格档位的总数量；Q |
| 73 | `cb_wall_imbalance` | `(bid_wall_size-ask_wall_size)/(sum)`，分母为零时取零；R |
| 74 | `log_ts_s` | 解析后的本地日志/发送时间戳；重设原点后的秒数 |

深度、斜率、距离和大额档位辅助函数在数据不可用时通常返回零；远端深度为零时，近远深度比返回一；没有成交量时，BUY 占比默认 0.5。距离方差为零时斜率返回零。“net add” 字段包含成交、撤单及价格区间移动造成的变化，不能单独代表新增限价单。事件计数反映经过频率限制的特征计算之间的变化，中间发生的交易所变化可能被遗漏。[L2 公式](../market_lib.py#L1490)。

EWMA 使用 `decay = exp(-dt/30)`、`var = decay*var + (1-decay)*r*r`，不除以经过时间。三十秒是衰减到 `1/e` 所需时间；经过时间非正时，代码把 decay 设为零。输出衡量的是事件间隔内的收益率波动，各间隔的时长可能不同。三秒已实现波动指标取窗口内各观测相对前一笔观测的收益率，累加其平方后取平方根。因此，某段收益率的起点可能早于窗口。[收益率特征](../market_lib.py#L534)。

<a id="b-preprocessing"></a>
#### B 归一化、时间与静态向量

每个资产的缓冲保留最近 32 个接入 tick。相邻时间戳倒退、缺失、非有限或间隔超过一秒时，会清空当前片段；相同时间戳可以保留。新连续片段积累到 32 个 tick 后才预测，不使用 padding。片段检查优先采用解析后的日志时间，其次是 `cb_last_ts`，最后是转为秒的服务器毫秒时间。检查在特征转型前使用 Python float 时间戳，因此间隔检查保留了 Python float 精度，模型特征则会经过下述 float32 转换。UTC 日内时间使用另行保存的绝对时间戳，无法取得时三个时间附加特征均置零。[片段检查](../quote_seq.py#L426)、[时间函数](../quote_seq.py#L687)。

接入时先分配 float32 行，再进行时间原点调整。当 `coinbase` 为正时，以 `_px` 结尾且不含 `vs_` 或 `bp` 的列，以及 `cb_spread_abs`，都会除以该价格；价格缺失则跳过。预测时，秒时间列减去窗口中最大的有限 `log_ts_s`，服务器毫秒时间先除以 1000 再减去同一基准。只有 `log_ts_s` 列不存在时才回退到 `cb_last_ts`。该列存在但所有值都缺失时，仍沿用原来的选择路径。2025 年 12 月附近的 float32 epoch 秒时间间隔已经达到 128 秒，因此减法之前亚秒级精度就已丢失。[接入](../quote_seq.py#L450)、[窗口组装](../quote_seq.py#L772)。

| 静态下标 | 内容 | z-score 前的单位 |
|---|---|---|
| 0 | 资产 ID | 以浮点数表示的整数类别 |
| 1–75 | 最新一行 75 个特征 | 与序列相同 |
| 76–150 | 窗口逐列均值 | 与序列相同 |
| 151–225 | 窗口逐列总体标准差 | 与序列相同 |
| 226 | UTC 日内秒数角度的正弦 | 无量纲 |
| 227 | UTC 日内秒数角度的余弦 | 无量纲 |
| 228 | 周末标记 | 0/1 |

静态摘要在重设时间原点之后、替换非有限值之前计算，因此一个缺失值可能使整列的均值/标准差摘要失效。随后先将序列和静态向量的非有限值替换为原始零，再按资产做 z-score。归一化后，原始零会变成 `-mean/std`。每个资产保存 75 个序列统计项和 229 个静态统计项。训练归一化对每个资产最多抽取两百万个序列行和静态窗口，用忽略 NaN 的总体统计量计算；极小或无效标准差替换为一，无效均值替换为零，也可以直接复用已有统计文件。[统计量](../train_seq.py#L135)、[数据集归一化](../train_seq.py#L259)。

<a id="b-network"></a>
#### B 网络层与输出解码

```text
(N,32,75) --> Linear 75->256 --> 正弦 PE + Dropout .1 --> 4 层 pre-norm 编码器
             --> last token (N,256)
(N,229) --> Linear 229->256 --------------------------------------------\
concat (N,512) --> Linear 512->512 --> GELU --> Dropout .1 --> Linear 512->256 --> GELU
               --> 四个 Linear 256->15 output heads 之一 --> (N,3,5)
               --> 各期限分别除以 [40000,30000,20000]
```

模型没有可学习的 asset embedding。资产 ID 既选择 output head，也进入静态向量。正弦 PE 是长度为 32、不写入 checkpoint 的 buffer。参数量为：序列投影 19,456；编码器 2,108,416；静态投影 58,880；body 393,984；四个 output heads 15,420；合计 2,596,156。实时路径会把非有限预测替换为零，同时仍可能将预测状态标记为成功。[网络](../train_seq.py#L397)、[实时预测](../quote_seq.py#L480)。

<a id="b-loss"></a>
#### B 损失、训练与指标

令 `e = clip(y*scale,±1000) - clip(q_pred,±1000)`，基础 Huber 函数为 `H_delta(e) = 0.5*min(abs(e),delta)^2 + delta*(abs(e)-min(abs(e),delta))`，delta 为 3。`e >= 0` 时 `a_k` 取分位数水平，否则取一减该水平；`c_k = [0.3,0.4,2.5,0.4,0.3]`；`w_nt` 为 NPZ 目标权重乘以资产权重，BTC 3、ETH 2、XRP 1、SOL 1。使用有限值 mask `m_ntk` 后，实际归约方式为：

```text
loss = sum_ntk(m_ntk * w_nt * c_k * a_k * H_3(e_ntk))
       / max(sum_ntk(m_ntk * w_nt), 1e-12)
```

五个分位数均有效时，分母是样本—目标权重总和的五倍。分位数权重不计入分母。这是非对称平滑损失；delta 为有限值时，其最优解可能偏离精确条件分位数。pinball loss 具有线性尾部和有界的残差梯度。[损失实现](../train_seq.py#L460)。

NPZ 分片提供 `x_seq`、`x_static`、`base_idx`、目标数组和权重。非有限标签替换为零，对应权重也置零；无效权重置零，但没有显式拒绝有限的负权重。各 worker 负责连续的分片子集，训练时打乱分片顺序，保留分片内部顺序。数据丢失，无法测量由此产生的 batch 内相关性。batch loss 非有限时跳过整个 batch。[数据集](../train_seq.py#L259)、[epoch 循环](../train_seq.py#L742)。

对于包含 `M` 个样本的分片，B 要求 `x_seq (M,32,75)`、`x_static (M,229)`、`base_idx (M,)`，以及各为 `(M,)` 的 `y_ret_2s`、`y_ret_5s`、`y_ret_10s`、`w_ret_2s`、`w_ret_5s`、`w_ret_10s`。加载器将目标/权重堆叠为 `(M,3)`，并使用 `allow_pickle=False`。归一化按文件名资产选择；如果保存的 `base_idx` 不一致，可能改变 output head 路由，却不改变归一化统计。构建器和实际分片已不存在，无法测试原分片是否一致。

保存的配置为 batch 8192、20 个 epoch、AdamW 学习率 `2e-5`、权重衰减 `3e-4`，逐 epoch 余弦退火到 `2e-6`，关闭 AMP，六个 worker，seed 42，不设步数上限。CLI 默认值则为 40 个 epoch、`1e-4` 和 `1e-2`。虽然解析了梯度裁剪参数，代码仍固定使用 1.0；B 会正确地先还原 AMP 梯度尺度再裁剪。验证按 batch loss 的平均值严格改善来选取模型，每个 batch 在模型选择中占相同权重；没有早停。随机种子覆盖 Python、NumPy 和 torch，但没有确定性 kernel 保证。[参数](../runs_seq_t/args.json)、[训练入口](../train_seq.py#L903)。

诊断包括加权 RMSE、R²、相关系数、方向准确率和经验分位数覆盖率。大幅变动诊断以四倍 sigma 定义事件，使用以预测中位数为中心、sigma 固定且来自元数据的正态分布。指标中的退化分母或非有限值可能被替换为零。仓库没有保存 B 的验证结果，无法据此确认预测质量。[指标](../train_seq.py#L526)。

<a id="b-policy"></a>
#### B 输出如何形成方向与报价偏移

策略对分位数的局部副本取累积最大值，强制其不下降。然后计算 `sigma_proxy = max(1e-12,(q75-q25)/1.349)` 和 `z = q50/sigma_proxy`；1.349 这一换算系数基于正态分布假设。代码通过线性插值估计零点 CDF，再取 `p_up = 1 - F(0)`，端点概率受已有 0.1/0.9 分位数限制。重复值和退化区间可能产生不可靠的概率估计；五个输出均为零时，辅助函数返回 0.9，但 `z=0` 会阻止常规信号激活。[策略辅助函数](../quote_seq.py#L138)。

```text
active_h = (p_up >= .58 or p_up <= .42) and abs(z) >= .20
score_h = sign(p_up-.5) * tanh(abs(z)/2) if active_h else 0
score = clip(.50*score_2s + .35*score_5s + .15*score_10s, -1, 1)
按 [2s,5s,10s] 顺序，若首个符合条件的期限满足 (q10>0 or q90<0) 且 abs(z)>=.60：
    score = 该期限的尾部方向（+1 或 -1）
delta_p = .02 * score
one_sided = abs(score)>=.80 或处于 tail-strong 模式
```

执行引擎将 `delta_p` 作为 `seq_skew`，却独立用 `abs(score)>0.1` 决定单边行为。策略给出的 0.80 标记未被交易引擎使用。条件式 crossing 阈值则为 0.5。另一个诊断适配器期望列表，但 `quote_debug` 实际为字典，导致汇总的 `quote_z`、`quote_mu`、`quote_sigma` 保持为零；这些字段不用于交易。[策略](../quote_seq.py#L535)、[适配器](../live_prediction.py#L123)。

---

<a id="model-c"></a>
### 模型 C：盘口中价残差与半价差

模型 C 读取混合的 Coinbase 与 Polymarket 事件。输出重建为 `mid_pred = mid_base + delta_mid`，再计算 `bid_pred = mid_pred - hs`、`ask_pred = mid_pred + hs`。预测残差并不能强迫模型学到有效结构，也不能排除接近恒等映射的解。历史目标的时间对齐方式仍未知；同时间戳 nowcast 与未来预测这两种解释都缺少验证依据。[训练损失](../train_iv.py#L498)、[实时引擎](../quote_iv.py#L391)。

<a id="c-inputs"></a>
#### C 输入：特征顺序与定义

顺序来自 [dataset_stats.json](../runs_iv_delta/dataset_stats.json)。每行延续当前事件未修改的字段；显式缺失更新则可能把原字段替换为 NaN。z-score 前，剩余非有限值变为原始零。所有数值通道，包括事件标记和 padding 行，都按资产标准化；之后 attention 用 mask 排除 padding key。均值/标准差取自 [norm_stats_iv.json](../data_iv64/norm_stats_iv.json)；实时路径把无效或不大于 `1e-6` 的标准差替换为一。

| 下标 | 特征 | 含义与原始单位 |
|---|---|---|
| 0 | `log_rel_px` | `log(coinbase/prev_close)`；moneyness（价内外程度）的近似指标，无量纲 |
| 1 | `cb_sigma_ewma` | Coinbase 事件收益率离散程度；无量纲 |
| 2 | `cb_rv_3s` | 三秒已实现收益率波动指标；无量纲 |
| 3 | `cb_buy_frac_1s` | 一秒 BUY 数量占比 |
| 4 | `cb_buy_frac_5s` | 五秒 BUY 数量占比 |
| 5 | `cb_flow1s_net` | 一秒 BUY 数量减 SELL 数量；标的币数量 |
| 6 | `cb_depth_imb_1bp` | Coinbase 1 bp 内深度失衡 |
| 7 | `cb_spread_bp` | Coinbase 价差；bp |
| 8 | `pm_iv_implied_900` | 经市场价格平滑修正的 900 秒 Student-t 尺度；`pm_iv_mult` 提供其校准乘子 |
| 9 | `model_iv` | 模型 A 的原始 900 秒尺度 |
| 10 | `model_df` | 预期为模型 A 自由度，实时接入未写入 |
| 11 | `tau` | 剩余秒数 |
| 12 | `pm_best_bid` | PM 买一价；美元/份 |
| 13 | `pm_best_ask` | PM 卖一价；美元/份 |
| 14 | `pm_mid` | 算术中价；美元/份 |
| 15 | `pm_spread` | 卖一价减买一价；美元/份 |
| 16 | `pm_size_bid_top` | 前五档买单份额之和 |
| 17 | `pm_size_ask_top` | 前五档卖单份额之和 |
| 18 | `pm_imb_top` | 预期为前五档数量失衡，生产端使用了不同键名 |
| 19 | `is_ticker_update` | 当前事件为 ticker；0/1 |
| 20 | `is_book_update` | 当前事件为 book；0/1 |
| 21 | `time_lag` | 最新事件时间减去本事件时间；非负秒数 |

下标 8 是第九个特征，下标 9 和 10 分别是第十、第十一个特征。实时路径没有按预期键名写入 `model_df` 和 `pm_imb_top`，后者实际写成 `pm_imbalance_top`。这两个通道先变为原始零，再通常变为非零的标准化常量。保存的非零均值确认其与归一化文件记录的数据分布不一致，但不能还原每个训练样本。实时路径已经写入 `tau`。[盘口摘要](../quote_iv.py#L152)、[ticker 接入](../quote_iv.py#L675)。

<a id="c-sequences"></a>
#### C 事件重放、mask 与基准中价

`_BaseSeq` 最多保留 4096 个按时间排序的事件。同时间戳采用右侧插入，保留到达顺序。顺序更新会追加一份延续已有字段的状态；迟到事件插入后，其后的全部状态重新按序回放。最后一个被淘汰的状态成为新 base state；早于该基准时间的事件直接丢弃。`time_lag` 和事件标记受保护，不能被任意稀疏更新覆盖。[事件存储](../quote_iv.py#L229)。

输入组装从最新事件向前遍历，取相邻间隔在零到一秒之间的连续片段。至少需要 16 个事件，最多取 64 个，右对齐放入初始为零的 `(64,22)` 数组，并生成布尔有效位置 mask。`time_lag` 在转为 float32 之前计算，避免了模型 B 的绝对时间戳精度问题。B 在连续片段中断时清空缓冲。C 选取最新的连续后缀，并可保留更早的历史。[输入组装](../quote_iv.py#L826)。

清理非有限值之前，`mid_base` 从后向前寻找最近一对有限、正值且 `ask >= bid` 的买卖价，再回退到严格位于 `(-0.25,1.25)` 内的中价。如果找不到参考值，解码仍可能以零为基准返回结果。即使事件时间是新的，延续下来的盘口价格也可能已经过期；代码不检查各字段距上次更新已过多久。当前训练代码则直接反标准化最后一行的 bid/ask，非有限值置零后构造基准中价。两套缺失值处理规则可能产生不同的参考中价。[实时参考值](../quote_iv.py#L858)、[训练参考值](../train_iv.py#L526)。

<a id="c-network"></a>
#### C 网络层、解码与损失

```text
(N,64,22) --> Linear 22->192 --> 正弦 PE + Dropout .1 --> 4 层 pre-norm 编码器
valid mask (N,64) --> src_key_padding_mask = NOT valid
last token (N,192) + Embedding(4,192)[asset_id]
    --> 所选资产的 output head：Linear 192->384 --> GELU --> Dropout .1 --> Linear 384->2
现有实时元数据：delta_mid = z0/100; hs = softplus(z1)/50
当前训练源码：  delta_mid = z0/100; hs = ReLU(z1)
```

PE 长度为 64，不写入 checkpoint。显式 asset embedding 在共享编码器之后相加，但不同资产的特征分布仍可能携带资产身份信息。参数量为：序列投影 4,416；编码器 1,188,096；embedding 768；四个 output heads 299,528；合计 1,492,808。构造函数默认宽度/FFN 为 256/512，CLI 和保存的张量则使用 192/384。[网络](../train_iv.py#L397)。

当前训练器按保存的顺序读取七列标签：bid、ask、mid、spread、买侧前几档数量、卖侧前几档数量、前几档失衡；仅优化 `Y[:,2]` 和 `Y[:,3]/2`。原始单位的标准差为 `s_mid=0.2581641412550873`、`s_hs=0.009014349689407113`，损失为：

```text
loss = mean(H_1((mid_base + z0/100 - mid_true) / s_mid))
       + .2 * mean(H_1((ReLU(z1) - half_spread_true) / s_hs))
```

没有样本权重或资产权重。batch loss 非有限时整批跳过，没有逐标签的有限值 mask。诊断包含中价/半价差 RMSE、残差摘要，以及预测买价高于标签卖价或预测卖价低于标签买价的比例。这些比例衡量预测与标签的关系。估计实际 crossing 成交率还需要标签时间、交易所延迟和订单提交信息。[损失与 epoch 指标](../train_iv.py#L498)。

<a id="c-compatibility"></a>
#### C 训练配置与兼容性

[保存的参数](../runs_iv_delta/args.json)为 batch 8192、32 个 epoch、学习率 `3e-4`、权重衰减 `5e-5`、六个 worker、seed 42、关闭 AMP、不设步数上限。当前 CLI 的学习率默认值不同，为 `1e-4`。按当前循环，AdamW 配合逐 epoch 余弦调度，会在 32 个 epoch 内把 `3e-4` 降到 `3e-5`。没有早停。梯度裁剪固定为 1.0；C 在 AMP 梯度还原尺度之前就裁剪，B 则先还原尺度。这一缺陷仅在 AMP 开启时生效，而保存的参数关闭了 AMP。[CLI](../train_iv.py#L42)、[epoch 循环](../train_iv.py#L594)。

特征统计使用训练数据中的有效序列位置，标签尺度使用总体标准差，也可以复用已有统计文件。当前训练器写出 `mid_std` 和 `half_spread_std`，而保存的解码文件使用 `mid_std_ref`、`half_spread_std_ref`、`MID_SCALE=100`、`HS_SCALE=50`、`spread_activation="softplus"`。实时路径依据该文件选择 `softplus(z1)/50`，当前训练损失却使用不带缩放的 ReLU。两套解码约定存在差异。历史训练的激活函数和源码版本先后仍未知。若不先统一约定，就把重新训练的权重与原有解码元数据配对，会产生风险。[统计量](../train_iv.py#L162)、[解码选择](../quote_iv.py#L445)。

按现存特征定义，C 要求 NPZ 数组 `X (M,64,22)`、`mask (M,64)`、`Y (M,7)`，资产由文件名推断。worker 分配和文件顺序打乱都保留文件内部样本顺序。最佳模型按验证损失严格改善选取。B/C 保存的日志间隔均为 50；B 的可选元数据/归一化路径覆盖项为 null，不强制重算归一化，两项采样上限均为 2,000,000。当前 C 加载器使用 `allow_pickle=True`，因此更不能加载不可信的替换数据集。C 的两份统计文件小数精度不同，但转为 float32 后，特征均值/标准差完全一致。

C 的实时输入来自 Coinbase 发送和 PM 价格变化回调，仅 PM 数量变化不直接触发。实盘使用外部预测请求合并器，关闭引擎可选的内部预测 worker。边界清理缓存不会等待已有 prediction drain task 结束，而 readiness 检查失败也不保证旧缓存失效。因此，即使名义预测间隔为 50 ms，缓存中仍可能保留过期结果。

---

<a id="quoting"></a>
## 5. 波动率校准、信号组合与报价

以下公式描述源码中的实现。价格采用概率单位，0.01 表示每份一美分。模型 B 的偏移上限为 0.02，模型 A 的直接修正上限为 0.01，模型 C 提供主要的报价中心与半价差。

<a id="calibration"></a>
### 根据市场价格校准波动率

校准器按名义 0.1 秒间隔遍历四个资产，采用 `p_obs = (bid*bid_qty + ask*ask_qty)/(bid_qty+ask_qty)`。该价格按买卖两侧各自的挂单量加权。前文定义的 Coinbase microprice 使用对侧挂单量加权。它要求正的挂单量、有效价格、参考价、原始模型 A 尺度/自由度和到期时间。对于不符合对称模型、落在 0.5 错误一侧的观测，代码跳过更新；可能原因包括模型与参考价不匹配，以及数据错误。[校准循环](../market_lib.py#L829)。

反解使用尺度区间 `[1e-5,5e-3]` 上的 Student-t 概率二分法，最多 50 次迭代，概率容差为 `1e-6`。超出区间或退化时不更新。moneyness 恰好为零时，概率恒为 0.5，无法识别尺度。偏离 0.5 也不保证数值条件一定良好。观测质量采用启发式规则：

```text
quality = 1/(1+(spread/.02)^2) * clip(2*abs(p_obs-.5),0,1)
y = log(clip(iv_observed/model_iv, .2, 5))
R = .04 / quality
gain = P / (P+R)
k_new = k + gain*(y-k)
P_new = (1-gain)*P + 1e-5
pm_iv_mult = exp(k_new)
pm_iv_implied_900 = pm_iv_mult * model_iv
```

正常初始状态为对数乘子零、方差 0.1；方差无效时，用当前观测和方差 0.04 重新初始化。按代码，过程方差在观测更新之后加入。模型 C 在下标 8 接收 `pm_iv_implied_900`。交易引擎需要 `pm_iv_mult` 来调整库存限制，概率计算使用模型 A 的原始尺度。修正尺度只在校准成功时更新，可能落后于新的原始尺度。校准器还依赖做市 worker 写入的 `coinbase_prev_close`。[滤波状态](../market_lib.py#L749)、[反解](../market_lib.py#L74)。

<a id="quote-center"></a>
### 报价中心与撤单价格带

交易引擎要求有限且为正的参考价、Coinbase 价格、原始 `model_iv`、`model_df`、`tau`、`pm_iv_mult`，有限的高低参考价格区间，严格为正的实际盘口价差，以及可用的 C 中价/半价差。它把 C 中价截断到 `[0,1]`，对半价差取绝对值，并在 `abs(delta_mid)>0.03`、`hs_pred>0.04`，或 `[pred_mid-2*hs_pred,pred_mid+2*hs_pred]` 与实际买卖价区间不相交时请求撤单。通过这些检查的预测仍可能有误差。[快照检查](../trade.py#L1207)。

低参考价向下、高参考价向上取到两位小数，XRP 则取四位。代码以当前标的价格 `S_now` 计算两个参考价对应的原始模型 A 概率，求平均后组合信号。在 `p_up(spot, barrier)` 中，尺度、自由度和剩余时间保持不变：

```text
t_fair_mid = .5 * (p_up(S_now, K_low_floor) + p_up(S_now, K_high_ceil))
center0 = clip(pred_mid + seq_skew, 0, 1)
iv_skew = clip(.15*(t_fair_mid-center0), -.01, .01)
fair_center = clip(center0 + iv_skew, 0, 1)
fair_yes_lo = clip(fair_center - .5*hs_pred, 0, 1)
fair_yes_hi = clip(fair_center + .5*hs_pred, 0, 1)
```

最后两个值定义撤单价格带。新订单的报价宽度按下节方法单独计算。YES 买单达到或高于下沿、卖单达到或低于上沿时成为撤单候选，NO 订单则使用互补值。实时定价没有实现 Avellaneda–Stoikov 库存偏移公式：`MMParams`、`DEFAULT_MM_BY_BASE`、`SPREAD_BPS`、`REF_PRICE`、`MM_MULT` 均未进入该路径，`rho` 仅用于日志。[中心计算](../trade.py#L1332)、[遗留配置](../trade_lib.py#L29)。

<a id="quote-width"></a>
### 宽度、tick 取整与条件式 crossing

附加宽度使用 `sigma_proxy = max(cb_sigma_ewma, cb_rv_3s/sqrt(3), 0)`。代码在 `prev_close` 附近把 `fair_center` 反解为标的价格，最多扩张区间 32 次、二分 48 次，然后对标的价格乘 `exp(±sigma_proxy)`，重新计算两点概率。概率差的一半加到 `hs_pred`，最终至少为一个 tick。这个近似量反映事件收益率的波动。一秒标准差的解释尚未验证，临近到期时的报价宽度也未必单调增大。反解失败时附加宽度回退为零。伪代码中的 `inverse_p_up` 将涨跌参考价固定为 `prev_close`，反解标的价格。[宽度](../trade.py#L1376)。

```text
S_center = inverse_p_up(fair_center, K=prev_close)
extra_half = .5 * max(p_up(S_center*exp(sigma_proxy), prev_close)-p_up(S_center*exp(-sigma_proxy), prev_close),0)
quote_half = max(hs_pred + extra_half, tick)
bid = clip(fair_center-quote_half,0,1)
ask = clip(fair_center+quote_half,0,1)
if abs(seq_score) <= .5:
    bid = min(bid, clip(pred_mid+hs_pred,0,1)-.01)
    ask = max(ask, clip(pred_mid-hs_pred,0,1)+.01)
```

引擎拒绝 `[0.01,0.99]` 之外的边缘价格，按 tick size 对买价向下、卖价向上取整，再应用检查，并尝试修复收缩到零或反向的价差。NO 价格由 `1-YES_ask` 和 `1-YES_bid` 推导，再向外取整。最终价格还受浮点运算和后续截断影响；盈利取决于后续成交和市场变动。[报价构造](../trade.py#L1494)。

crossing 限制有条件地生效，并以预测盘口为参考。订单到达交易所时，实际 best bid/ask 可能已经不同。提交使用 GTC，没有显式 post-only 标记。可立即成交的 GTC 限价单可以作为 taker 成交，剩余数量继续挂单。交易所强制执行的 post-only 保护需要另行设置订单标记。官方 [Polymarket 订单生命周期说明](https://docs.polymarket.com/concepts/order-lifecycle)解释了这些订单语义。归档客户端的历史行为仍未验证；现有提交路径允许 taker 成交。[GTC 提交](../trade_lib.py#L1160)。

---

<a id="execution"></a>
## 6. 执行、库存、并发与关停

<a id="order-lifecycle"></a>
### 本地订单生命周期与交易所状态的不确定性

```text
快照 --> 允许资产 / token 查询 --> 最新待处理版本 --> 每个 YES token 的 worker
worker --> 报价与风险过滤 --> 从本地记录移除撤单候选
                         |--> detached 撤单请求 --> 成功 / 失败 / 未知
                         \--> 容量检查 --> detached GTC 提交 --> 响应 / 超时
响应 --> 已接受的 ID --> 本地记录 --> 可选的旧版本订单撤单检查
用户 MATCHED 事件 --> 库存增量（本地活跃订单剩余数量保持不变）
关停 / 超时无输入 / 收尾 --> 尝试撤单（交易所仍可能有订单或持仓）
```

`on_tick` 通过 `TokenIndex` 解析 slug，记录本地活动时间，覆盖最新待处理快照。每个 YES token 的 worker 持续处理期望版本，直到追上最新版本。同步入口可能因文件系统中的 token 索引刷新而阻塞。`_quote_and_trade_atomic` 中的“atomic”描述本地报价周期的组织方式；交易所分别处理由此产生的请求。[入口](../trade.py#L784)、[worker 调度](../trade.py#L820)。

撤销报价已过时或方向相反的订单时，代码先修改本地记录，再发出 fire-and-forget 网络请求。程序会等待为释放容量发起的撤单请求结束；封装函数未根据响应内容核实每笔订单是否已在交易所撤销。撤单与提交可能重叠。两毫秒“提交后等待”从任务调度完成后开始，可能在交易所确认之前结束。即使本地记录为空或日志出现 `done`，交易所仍可能有活跃订单。[撤单](../trade.py#L953)、[容量管理](../trade.py#L1030)。

在途提交记录版本号和 stale 标记。新的提交尝试可能把旧任务标为 stale，并丢弃新批次；旧任务收到响应后，再按最新撤单价格带检查。新报价可能绕过这一调度路径，因此旧报价仍可能留在盘口。代码从响应列表中过滤出已接受订单的 ID，再按位置与原始订单配对；若前一笔被拒而后一笔接受，接受的 ID 可能对应到错误的价格或 token 元数据。[提交状态](../trade.py#L616)、[ID 配对](../trade.py#L572)。

I/O 默认使用八个线程、容量为八的信号量、两次尝试，以及 `0.05*(1+0.5*random())` 的随机退避。外层 0.8 秒 asyncio 超时覆盖信号量等待和重试，却不能停止已运行的阻塞线程。手写签名请求没有 HTTP 超时。提交是否成功不明确时，重试会再次创建和签名订单，可能造成重复头寸；没有应用层幂等机制解决这种不确定性。日志 `salt` 只是关联键。[I/O](../trade.py#L276)、[签名请求](../trade_lib.py#L373)。

容量默认值为每 slug 每秒一个提交批次、每侧四笔活跃订单、相同 token/方向/价格最多两笔。“侧”按 BUY 或 SELL 汇总两个 token 的订单；YES 方向暴露还取决于交易的是哪一个 token。容量不足时，先撤销在 YES 价格空间中距离公允中心最远的订单；距离相同则先撤较新的订单。提交频率在检查中转为整数，介于零与一之间的正值可能阻止全部提交。成交后，本地订单记录的剩余数量保持不变，因此容量估计可能偏离交易所状态。

<a id="inventory"></a>
### 库存过滤与持仓追踪

本地库存根据用户 WebSocket 的 `MATCHED` 增量估计，从零开始，遗漏账户已有持仓。BUY 增加份额，SELL 减少份额。归属判断同时检查顶层 owner 和匹配的 maker 条目；消息指纹保存在有界 FIFO 历史中去重，默认 20,000 条。结算失败或后续状态改变时，代码不会据此修正这套库存记录。持仓 API 轮询独立检查上限，并保持本地库存估计不变。[成交追踪](../trade_lib.py#L771)。

| 资产 | 报价数量 | 库存上限 | 每 slug 累计成交份额上限 |
|---|---|---|---|
| BTC | 5 | 20 | 20,000 |
| ETH | 5 | 15 | 2,500 |
| SOL | 5 | 15 | 1,500 |
| XRP | 5 | 15 | 1,500 |

四种资产都读取 `MM_BTC_QSIZE`、`MM_BTC_INVCAP`、`MM_BTC_VOLCAP`，设置其中一个就会影响所有资产。实际订单数量还要满足市场规定的最小下单份额。默认动作是在买价买 YES、在卖价的互补价格买 NO。如果估计的 YES 持仓超过实际报价数量的五倍，就把买 NO 改为卖 YES；NO 持仓超限时反向处理。系统没有完整的可用余额或在途订单预留计算，不能保证固定数量的卖单有足够持仓支持。[配置](../trade_lib.py#L69)、[动作](../trade.py#L1630)。

令 `net = YES_balance - NO_balance`、`C = inv_cap`、`a = 1-clip(tau,0,900)/900`、`d = sign(coinbase-prev_close)*sign(pm_iv_mult-1)`，则库存区间为：

```text
d > 0: [C*(1.2*a-1), C]
d < 0: [-C, C*(1-1.2*a)]
d = 0: [-C, C]
```

代码仅在当前净库存严格越界时，过滤继续扩大该方向暴露的动作。它不为在途订单预留额度，也不强制限制成交后的预计库存，更无法强迫订单成交或账户平仓。未超限时，模型 B 的方向非 flat 且 `abs(score)>0.1` 会过滤新动作；超限时，新动作优先服从风险过滤。此前针对反向挂单的撤单检查仍是独立步骤。[库存策略](../trade.py#L1669)。

<a id="timing"></a>
### 时间限制与风控线程

```text
窗口起点 S                                         到期 S+900
S .. S+8          开始阶段限制新订单
S+8 .. S+885      名义报价区间（877 秒，还需通过其他检查）
S+885 之后       每个窗口起点触发一次账户级收尾撤单尝试
约每 .10 秒      检查是否连续 .5 秒没有本地快照
约每 5 秒        独立检查账户持仓上限
```

收尾分支使用账户级撤单，包括当前 slug 以外的订单；较早的输入校验失败也可能使函数在到达该分支前返回。inactivity dead-man switch 检查本地可接受快照的到达时间。这些快照可能携带过期的交易所数据，撤单仍是尽力执行。TTL 默认关闭，注释将原因写为保留排队优先级，但仓库没有量化证据证明排队优先级对盈利的贡献。[时间限制](../trade.py#L474)、[风控循环](../trade.py#L427)。

持仓线程每五秒请求一次，超时 2.5 秒，最多返回 500 项、offset 为零、不分页。它检查当前索引中的四种资产 token，即使交易引擎只做 BTC。某 token 持仓超过对应 cap 的四倍时调用 `os._exit(2)`；连续三次请求失败或无法确定账户地址时调用 `os._exit(3)`。这些强制退出前不撤单，原有挂单可能继续存在。token 映射为空时会跳过检查，账户余额将处于未检查状态。[持仓检查](../trade_lib.py#L528)。

用户 WebSocket 在独立线程中运行，初次连接后最多再重试两次，间隔一秒。健康检查线程判断：距最近一次提交是否超过五秒，且此后没有用户消息。不断提交新批次可能推迟该条件成立。线程存活时，认证订阅仍可能处于等待就绪的阶段。soft-death 处理会清空持仓和会话成交量估计，却保留去重历史；交易引擎在断线后仍保留成交量最大估计。后续成交量仍可能被低估。[WebSocket](../trade_lib.py#L927)、[soft-death](../trade_lib.py#L1336)。

<a id="shutdown"></a>
### 并发与关停限制

库存回调在 WebSocket 线程执行，先修改待处理快照和版本计数，再调度回主循环。这些复合状态更新违反了状态只由主事件循环修改的约定，需要 GIL 之外的同步保护。两份独立刷新的 `TokenIndex` 和共享 HTTP session 还会带来条件性的状态一致性风险，其实际发生情况尚未复现。[回调](../trade.py#L647)。

关停会禁用交易、尝试逐 slug 及账户级撤单、清空本地映射，却不等待所有 worker、detached 提交/撤单任务或底层 I/O 线程结束，也没有完整关闭 session/executor 或注销全部回调。服务在取消生产者之前先调用这套关停逻辑，已经运行的请求可能随后完成。校准器任务句柄未保存在显式关停列表中，但 `asyncio.run` 通常会在循环退出时取消剩余任务。[引擎关停](../trade.py#L486)、[服务清理](../live_prediction.py#L1565)。

引擎禁用时，inactivity 循环仍存活并等待；恢复时又会启动一个循环，可能造成重复。延迟重连也缺乏完整的停止和等待退出协议。结合强制退出及网络确认不明确等情况，关停完成并不能证明账户已经没有活跃订单或持仓。

---

<a id="operations"></a>
## 7. 配置、依赖、日志与排错

<a id="configuration"></a>
### 配置参考

下表列出源码默认值；训练 CLI 默认值和保存的运行参数见各模型章节。认证信息省略真实取值。`TradeSession` 初始化时，从进程工作目录读取 `key.env`，按简单的 `KEY=VALUE` 格式逐行解析，并使用 `os.environ.setdefault`。已有环境变量优先，引号会作为字面值保留，shell `export` 和多行语法不受支持。[解析器](../trade_lib.py#L113)。

覆盖配置时需要注意加载顺序。`BASE_CFG` 中的资产限制、模块级订单与时间设置，以及 `live_prediction.py` 中的模型开关、路径和预测间隔，都在导入模块时读取。在实盘入口中，`TradeEngine` 也会先读取 I/O 设置，再创建负责加载 `key.env` 的会话。覆盖这些值需要事先将变量放入进程环境；随后加载文件时，已经初始化的配置值保持不变。`PM_OWNER_ID`、`WS_SEEN_LIMIT` 等会话设置则在文件加载后读取。[资产限制](../trade_lib.py#L69)、[订单设置](../trade.py#L52)、[模型设置](../live_prediction.py#L59)、[引擎初始化](../trade.py#L155)、[会话初始化](../trade_lib.py#L261)。

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `PRIVATE_KEY` / `PM_PRIVATE_KEY` / `PK` | 必需，按顺序取首个可用值 | 签名私钥 |
| `PM_FUNDER` / `FUNDER` | 未设置 | 出资地址；存在时使用签名类型 1，否则 0 |
| `PM_ADDRESS` | 未设置 | 持仓检查的备用地址 |
| `PM_OWNER_ID` | 派生 API key | 覆盖成交归属 owner |
| `PM_TEMP_DIR` | `temp` | session 的 token 索引目录 |
| `WS_SEEN_LIMIT` | 20000 | 撮合事件去重容量 |
| `MM_BTC_QSIZE` | 5 | 所有资产的报价数量 |
| `MM_BTC_INVCAP` | BTC 20，其他 15 | 所有资产的库存上限 |
| `MM_BTC_VOLCAP` | 20000 / 2500 / 1500 / 1500 | BTC / ETH / SOL / XRP 累计成交份额上限 |
| `MM_POST_SUBMIT_WAIT_S` | .002 | 调度提交后的等待 |
| `MM_ORDER_TTL_S` | 0 | 单笔订单 TTL，零表示关闭 |
| `MM_TTL_POLL_INTERVAL_S` | .25 | TTL 扫描间隔 |
| `MM_ORDER_TTS_CANCEL_S` | .5 | 本地快照中断阈值 |
| `MM_INACTIVITY_POLL_S` | .10 | 检查间隔，最小 .05 |
| `MM_MAX_SUBMITS_PER_SEC` | 1.0 | 每 slug 批次频率，检查时转整数 |
| `MM_MAX_LIVE_PER_SIDE` | 4 | 本地记录中每 BUY/SELL 侧的活跃订单上限 |
| `MM_MAX_SAME_PRICE` | 2 | 每 token/方向/价格的本地上限 |
| `MM_IO_EXEC_WORKERS` | 8 | 阻塞 I/O 线程池 |
| `MM_MAX_PARALLEL_IO` | 8 | I/O 信号量容量 |
| `MM_IO_TIMEOUT_S` | .8 | 外层异步超时 |
| `MM_IO_RETRIES` | 2 | I/O 总尝试次数 |
| `MM_IO_BACKOFF_S` | .05 | 随机重试等待的基数 |
| `ENABLE_SEQ` | 1 | 启用模型 B |
| `SEQ_RUN_DIR` | `runs_seq_t` | B checkpoint/配置目录 |
| `SEQ_DATA_DIR` | `data_seq32` | B 特征/归一化元数据 |
| `SEQ_CKPT` | `model_best.pt` | B checkpoint 文件名 |
| `SEQ_PRED_MIN_INTERVAL_MS` | 50 | B 每资产预测间隔 |
| `ENABLE_IV` | 1 | 启用模型 C |
| `IV_RUN_DIR` | `runs_iv_delta` | C checkpoint/解码目录 |
| `IV_DATA_DIR` | `data_iv64` | C 归一化元数据 |
| `IV_PRED_MIN_INTERVAL_MS` | 50 | C 每资产预测间隔 |
| `IV_ROLL_CLEAR_EPS_SEC` | 1 | 边界后 C 清理延迟 |
| `CB_MISSING_LOG_GRACE_SEC` | 2.0 | 缺失数据告警的启动宽限期 |
| `CB_MISSING_LOG_RATE_SEC` | 2.0 | 缺失数据告警间隔 |
| `CB_L2_FEATURE_INTERVAL_S` | .10 | L2 重算间隔 |
| `CB_ROTATE_MAX_BYTES` | 67108864 | 启用原始 Coinbase 写盘时的轮转大小 |
| `CB_FLUSH_EVERY` | 200 | 启用原始 Coinbase 写盘时的刷新频率 |
| `LIVE_LOG_STDOUT` | 1 | 主日志 stdout 镜像 |

交易引擎需要 C 输出的中价/半价差，关闭 C 会使报价流程缺少这些输入。部分路径和资产列表写死在代码中。认证流程会初始化 Polygon chain ID 137 的 CLOB client，并通过交易场所派生 API 认证信息；启动实盘入口会发起需要认证的网络请求。[Session 初始化](../trade_lib.py#L245)。

<a id="dependencies"></a>
### 依赖与 checkpoint 加载

导入依赖包括 PyTorch、NumPy、pandas、SciPy、scikit-learn、matplotlib、aiohttp、websockets、websocket-client、requests 和 py-clob-client。仓库没有依赖锁文件或经过测试的环境规格。带 slots 的 dataclass 等语法意味着正常运行至少需要 Python 3.10。兼容性仍取决于具体依赖版本。训练代码支持 CPU/CUDA 选择。

模型 A 显式调用 `torch.load(..., weights_only=False)`，允许 pickle 执行。B/C 没有传入该参数，因此行为取决于版本：PyTorch 文档说明，从 2.6 起，未指定 `pickle_module` 时默认 `weights_only=True`，更早版本默认较宽松。受限加载仍存在安全风险。请勿替换为不可信来源的 checkpoint。[A 加载器](../live_lib.py#L238)、[B 加载器](../quote_seq.py#L350)、[PyTorch 序列化文档](https://docs.pytorch.org/docs/main/notes/serialization.html)。

<a id="logging"></a>
### 日志与运行时文件

主日志每行由时间戳/级别前缀和 JSON 组成，解析时需要先处理前缀，再解码 JSON 内容。格式化器可能生成重复或位置不正确的 UTC `Z` 标记。轮转大小为 10,000,000 字节，最多 4096 个备份。每次决策的 `salt` 用于关联预测、报价和订单日志；提交重试路径仍缺少幂等协议。`hub_debug.py` 每秒以 JSONL 格式写入公开属性快照，排除私有成交缓冲、EWMA 和校准状态。代码中没有实际终端仪表盘，`refresh_event` 也没有消费者。[日志器](../live_lib.py#L53)、[hub 快照](../hub_debug.py#L1)。

| 运行时路径 | 内容 |
|---|---|
| `data/logs/live_predictor.log` | 带时间戳/级别前缀的 JSON 事件 |
| `data/debug/hub-NNNNN.jsonl` | hub 公开属性快照，50 MB 轮转 |
| `data/polymarket/resolution/*.jsonl` | Chainlink 边界记录 |
| `data/polymarket/rtds-other.jsonl` | 其他 RTDS 消息 |
| `temp/{slug}.json` | 发现的市场元数据 |
| `logs/ws_user/sent/` | 订阅消息中的明文 API key、secret、passphrase |
| `logs/ws_user/recv/`、`logs/ws_user/matches/` | 用户频道事件与撮合成交 |
| `logs/inv_watch/` | 钱包地址、token ID、账户持仓 |

<a id="troubleshooting"></a>
### 只读排错参考

| 现有日志中的现象 | 可对照的源码原因 |
|---|---|
| 没有 A 预测 | 缓冲不足 300 行；缺少分钟行；reseed；checkpoint/scaler 不兼容 |
| `waiting_prevclose_coinbase` | 缺少本地边界前历史，启动时没有回填 Coinbase |
| `skip_cb_tick_pred_no_fresh_model` | A 预测缺失或超过 20 秒 |
| `skip_stale_pm_ts` | 该资产 PM 数据超过 10 秒；订阅空档或盘口陈旧 |
| B/C 未就绪 | B 需要 32 个连续 tick；C 需要至少 16 个事件，相邻间隔不超过一秒 |
| C 中价增量/半价差被拒 | 输出超过 .03/.04 阈值，或预测区间与实际买卖价区间不相交 |
| 撤单成功日志后交易所仍有订单 | 本地先移除；响应失败/不明确；还有在途提交 |
| 诊断异常为零 | B 的 `quote_debug` 类型不匹配；交易方向信号单独计算 |
| 记录生成后出现 `resolution` 错误 | Chainlink 日志重复传入 `event` |
| 进程直接消失 | 持仓风控调用 `os._exit`；检查本地日志时不要公开认证信息 |

本节用于离线解释已有日志。当前交易所 API 的兼容性仍未验证。

---

<a id="training"></a>
## 8. 训练流程、模型文件与复现限制

<a id="artifacts"></a>
### 模型文件清单

| 模型文件 | ZIP 成员 / storages | State 元素数 | 可训练参数 |
|---|---|---|---|
| `runs/BTC/artifacts/model_checkpoint.pt` | 85 / 79 | 814,277 | 420,995 |
| `runs/ETH/artifacts/model_checkpoint.pt` | 85 / 79 | 814,277 | 420,995 |
| `runs/SOL/artifacts/model_checkpoint.pt` | 85 / 79 | 814,277 | 420,995 |
| `runs/XRP/artifacts/model_checkpoint.pt` | 85 / 79 | 814,277 | 420,995 |
| `runs_seq_t/model_best.pt` | 70 / 64 | 2,596,156 | 2,596,156 |
| `runs_iv_delta/model_best.pt` | 73 / 67 | 1,492,808 | 1,492,808 |

保存的张量 storage 均使用 float32，张量值均为有限值。可训练参数合计 `4*420995 + 2596156 + 1492808 = 5772944`。仅凭张量结构不能确认模型推理能够成功运行。B/C checkpoint 为 state dictionary，架构和解码还依赖外部 JSON。A 则包含模型配置、scaler、日期范围、epoch 摘要、选中 epoch 的指标、PIT 校准器和保存时间戳。

<a id="saved-results"></a>
### 保存的 A 结果与数据日期

全部 A checkpoint 记录的训练日期为 2025-02-14 至 2025-12-14，测试日期为 2024-12-01 至 2025-02-13。该评估使用早于训练期的测试数据。每个选中 epoch 的指标记录训练 4,198,319、验证 1,036,795、测试 1,295,282 个样本。四个文件均于 2025 年 12 月 16 日保存。下表列出 checkpoint 元数据中保存的指标，并对小数位作了取舍。

| 资产 | 选中 epoch | 验证 NLL | 保存的 `ece` | KS D | `iv` 均值 | `df` 均值 |
|---|---|---|---|---|---|---|
| BTC | 6 | -5.553313 | .000490308 | .0106356 | .001499341 | 6.824050 |
| ETH | 6 | -5.022279 | .000246560 | .0061785 | .002364110 | 5.951548 |
| SOL | 7 | -4.792816 | .000575636 | .0176308 | .003028617 | 7.614213 |
| XRP | 8 | -4.935689 | .000498811 | .0091836 | .002699561 | 7.406803 |

由于改进阈值为 `1e-4`，选中 epoch 不一定具有精确到末位小数的最低 NLL。连续密度在较小收益率单位下出现负 NLL 是正常的。保存的 KS p 值很小。时间相关、缺失的数据构建过程和已发现的特征时间问题，限制了这些数值及非标准 `ece` 的解释范围。未来校准质量与交易表现仍需单独验证。

<a id="dataset-metadata"></a>
### B/C 元数据与训练产物

模型 B 元数据列出 3019 个训练分片和 756 个验证分片，分别有 6,574,548 和 2,123,004 个样本。分片按 slug 划分，各资产的验证数据时间均晚于训练数据。history-root 标签引用六个采集目录，但目录名称本身不足以证明采集覆盖连续。权重元数据记录 `power_law_per_base`、`z0=1`、`lambda=1`、`p=1.5`、`wmax=16`，实际构建公式已不可得。

| 目标 sigma | BTC | ETH | XRP | SOL |
|---|---|---|---|---|
| `ret_2s` | .00011779 | .00016510 | .00018418 | .00018367 |
| `ret_5s` | .00018863 | .00026412 | .00029160 | .00028869 |
| `ret_10s` | .00026799 | .00037589 | .00041408 | .00040852 |

C 的归一化和解码文件保留了特征/标签名及尺度，但没有完整数据集或验证结果。当前训练程序会保存参数、归一化、epoch 日志、含优化器的 epoch checkpoint 和最佳模型状态；大多数相应历史产物已经不存在。`args.json` 保存运行配置。

<a id="reproducibility"></a>
### 无法复现或确认的内容

缺少原数据集、构建器和 A 训练入口，无法重新生成六个历史模型。仓库无法证明 B/C 精确标签时间、采集完整性、历史依赖版本、新市场环境下的模型质量、成交、盈亏、手续费影响或实测延迟。除张量结构外，预处理兼容性、解码兼容性和部署就绪情况均需分别验证。按实时源码重新实现构建器，将得到采用自身构建规则的新实验数据集。

仍未厘清的历史训练细节包括：B 是否经过微调、C 的剩余期限采样分布、源码版本先后，以及训练时使用的完整 shell 命令和源码版本。在这些边界内，现存代码仍可用于研究实现选择和故障模式。

---

<a id="issues"></a>
## 9. 已验证问题、实现限制与安全

<a id="security"></a>
### 发布与运行安全

WebSocket 订阅消息会把派生 API key、secret 和 passphrase 写入 `logs/ws_user/sent/`；持仓检查会把钱包和账户持仓写入 `logs/inv_watch/`。这些文件在运行时生成。当前目录中没有这些运行日志。owner ID 从 `PM_OWNER_ID` 或派生 API key 获取。[订阅记录](../trade_lib.py#L967)、[持仓快照](../trade_lib.py#L631)。

`.gitignore` 排除了 `data/`、`temp/`、`key.env` 和 `logs/`，但没有完整覆盖各种环境文件名、私钥文件、缓存及其他目录内的输出。忽略规则不保护压缩包，也不能保护已被追踪的文件。checkpoint 和训练参数可能包含元数据与路径。发布前应检查认证文件与生成的账户数据。宽松 checkpoint 加载见[依赖](#dependencies)。

<a id="issue-catalog"></a>
### 问题与实现限制清单

下表按类别汇总缺陷、条件性风险、配置与兼容性约束，以及指标定义上的限制。类别用于标明各项性质。“已确认”表示行为可由源码检查确定，其财务影响仍未验证。每项均链接到相应的技术说明与证据。

| 编号 | 类别 | 行为、影响与证据 |
|---|---|---|
| 1 | 未使用配置 | `MMParams` 不参与报价，`rho` 仅记录日志。[中心](#quote-center) |
| 2 | 已确认缺陷 | 所有资产读取 BTC 命名的配置变量。[库存](#inventory) |
| 3 | 文档不一致 | 持仓检查使用 cap 的 4 倍；注释写为 3 倍，低估了实际阈值。[时间限制](#timing) |
| 4 | 条件性安全风险 | 强制退出绕过撤单，已有挂单可能继续存在。[时间限制](#timing) |
| 5 | 兼容性回退 | `buy/sell` 回退到 score 符号判断，未发现由此导致的方向反转。[B 策略](#b-policy) |
| 6 | 已确认诊断缺陷 | 字典/列表不匹配使 B 汇总诊断保持零。[B 策略](#b-policy) |
| 7 | 已确认输入缺陷 | `pm_imbalance_top` 未写入 `pm_imb_top`；z-score 前为原始零。[C 输入](#c-inputs) |
| 8 | 已确认输入缺陷 | 实时接入遗漏 `model_df`；z-score 前为原始零。[C 输入](#c-inputs) |
| 9 | 条件性正确性缺陷 | 批次部分接受时，接受 ID 可能与原订单元数据错配。[订单](#order-lifecycle) |
| 10 | 条件性安全风险 | 不明确的提交/重试可能重复头寸，没有解决此问题的幂等协议。[订单](#order-lifecycle) |
| 11 | 策略规则差异 | 交易引擎的 >.1 阈值可能在 B 的 .8 标记生效前触发单边行为。[库存](#inventory) |
| 12 | 需限定的记账风险 | soft-death 清空 session 计数，引擎保留此前的最大值；后续计数可能低于实际成交数量。[时间限制](#timing) |
| 13 | 运行限制 | 没有专门的 429 处理策略；重试行为因请求路径而异，部分非 200 响应会立即返回。[数据源](#feeds) |
| 14 | 条件性并发风险 | 共享可变 HTTP session 没有显式逐请求同步，未复现故障。[关停](#shutdown) |
| 15 | 条件性一致性风险 | 两份 token 索引独立刷新。[关停](#shutdown) |
| 16 | 策略范围风险 | 四资产持仓检查可能终止仅交易 BTC 的会话。[时间限制](#timing) |
| 17 | 已确认状态缺陷 | 新 snapshot 到达前，PM 盘口保留旧合约状态。[数据源](#feeds) |
| 18 | 生命周期限制 | 校准器未纳入显式清理，但事件循环退出仍可能取消它。[关停](#shutdown) |
| 19 | 未使用信号 | `refresh_event` 无消费者，发出该信号不会触发仪表盘更新。[日志](#logging) |
| 20 | 已确认配置缺陷 | 两个训练器忽略 `--grad-clip-norm`，固定使用 1.0。[B 损失](#b-loss)、[C 兼容性](#c-compatibility) |
| 21 | 条件性训练缺陷 | C 在 AMP 下裁剪仍带缩放的梯度；保存参数关闭 AMP。[C 兼容性](#c-compatibility) |
| 22 | 默认目标限制 | A 默认概率乘子不随预测变化，是否严格抵消取决于截断/epsilon。[A 训练](#a-training) |
| 23 | 指标区别 | 训练优化加权 NLL，验证衡量未加权 NLL，数值反映了不同的权重设置。[A 训练](#a-training) |
| 24 | 未使用常量 | forward 不应用固定 `IV_FLOOR`。[A 网络](#a-network) |
| 25 | 文档不一致 | A 的输出尺度系数为 .0005；docstring 写为 .0004。[A 网络](#a-network) |
| 26 | 未使用模型产物 | 保存的 PIT 映射不用于实时路径，市场校准是独立机制。[A 实时行为](#a-runtime) |
| 27 | 已确认约定不一致 | 当前 C 训练与现存实时元数据使用不同半价差解码，版本先后未知。[C 兼容性](#c-compatibility) |
| 28 | 兼容性约束 | B/C 的 SOL/XRP ID 对调，但各引擎正确使用各自映射。[模型](#models) |
| 29 | 训练顺序限制 | 无分片内打乱；数据缺失，无法测量实际相关性。[B 损失](#b-loss) |
| 30 | 训练数据风险 | A 现有辅助函数按分钟开盘索引选取完整分钟特征，存在分钟内前视；历史训练入口未知。[A 训练](#a-training) |
| 31 | 实时数据风险 | 未收盘的 REST K 线可能被固定为实时分钟特征：代码不检查收盘时间，且只在开盘时间晚于已有记录时插入。[A 实时行为](#a-runtime) |
| 32 | train/serve skew | A 的实时秒序列省略空秒并接受倒退时间戳，历史输入则使用完整网格，存在时间对齐差异。[A 实时行为](#a-runtime) |
| 33 | 已确认精度缺陷 | B 在重设时间原点之前丢失精度；保存时期附近的 float32 epoch 秒间隔为 128 秒。[B 预处理](#b-preprocessing) |
| 34 | 指标定义限制 | A 的 `ece` 衡量箱内均值偏移；即使 PIT 不均匀，所有观测集中在某个分箱中点时误差也可为零。[A 实时行为](#a-runtime) |
| 35 | 条件性订单状态风险 | 撤单确认前移除本地记录，加上响应解析和成交核对不完整，可能使撤单记录与交易所状态不一致。[订单](#order-lifecycle) |
| 36 | 条件性执行风险 | 异步超时或关停不会停止阻塞式提交；在途线程可能在本地超时或关停后完成提交。[订单](#order-lifecycle)、[关停](#shutdown) |
| 37 | 持仓记账限制 | 持仓估计遗漏启动余额和结算核对，可能偏离账户实际持仓。[库存](#inventory) |
| 38 | 条件性并发风险 | 库存回调从其他线程修改事件循环管理的状态，对待处理快照和版本的复合更新没有同步保护。[关停](#shutdown) |
| 39 | 时效性检查限制 | heartbeat 可能重复已经过期的 Coinbase 数据；字段与缓存的有效期未得到全面检查。[共享状态](#shared-state) |
| 40 | 条件性状态风险 | 合约切换时的清理不取消模型 C 在途的 prediction drain task，任务仍可能把上一份合约的结果写回缓存。[共享状态](#shared-state) |
| 41 | token 映射一致性风险 | token 选择路径不同：CLOB 读取器丢弃 outcome，执行索引则保留。[数据源](#feeds) |
| 42 | 输入更新限制 | 仅 PM 挂单量变化时不触发 C 盘口事件接入，hub 更新与 C 事件流可能不同步。[数据源](#feeds) |
| 43 | 条件性日志缺陷 | Chainlink 日志调用通过位置参数和关键字重复传入 `event`，可能在状态或文件写入后报错。[参考价格](#reference-prices) |
| 44 | 生命周期风险（有条件触发） | 恢复运行可能重复启动 inactivity 循环：禁用后的循环仍存在，恢复时又创建额外实例。[关停](#shutdown) |
| 45 | 模型与加载兼容性风险 | 回退路径可能选错模型 A 的资产或省略 scaler；数据加载器的 prefetch 设置也存在零 worker 时的兼容性约束。[A 实时行为](#a-runtime)、[A 训练](#a-training) |
| 46 | 订单与风险控制限制 | 仍可能出现 taker 成交或成交后库存越界：crossing 检查使用预测盘口，库存过滤使用当前余额。[宽度](#quote-width)、[库存](#inventory) |
| 47 | 无效输出处理风险 | 无效模型输出可能变成表面可用的结果：B 会处理非有限输出，C 可能使用零参考价解码。[B 网络](#b-network)、[C 序列](#c-sequences) |

仍可能存在其他问题。各项说明代码行为及其限制，与历史交易损失的因果关系尚未验证。

<a id="license"></a>
### 许可证与适用范围

[Sustainable Use License 1.0](../LICENSE) © QuantumSlayer。本项目以源码可用（source-available）的方式发布，许可条件见[项目 README](../README.zh-CN.md#许可证)，以完整英文许可证为准。本手册记录归档代码，供技术研究使用。系统已停止维护，连接有资金的账户存在安全风险。当前交易所行为、参与资格规则和兼容性均未测试。
