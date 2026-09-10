# 技术文档

[English version](README.md) · [项目 README](../README.zh-CN.md)

技术手册介绍 poly-15min 的合约、数据处理、模型、定价和订单执行，适合熟悉 Python、量化交易和机器学习的读者。

**[完整技术手册](technical-reference.zh-CN.md) · [模型](technical-reference.zh-CN.md#models) · [报价](technical-reference.zh-CN.md#quoting) · [安全](technical-reference.zh-CN.md#security)**

---

## 章节导航

| 章节 | 内容 |
|---|---|
| [1. 项目状态与历史](technical-reference.zh-CN.md#status) | 归档状态、项目历史，以及现存源码和模型文件 |
| [2. 合约与定价假设](technical-reference.zh-CN.md#contracts) | 二元兑付、参考价格、剩余期限和 Student-t 概率 |
| [3. 运行流程与行情](technical-reference.zh-CN.md#runtime) | 源码结构、异步任务、数据源、共享状态和合约切换 |
| [4. 三个 Transformer 模型](technical-reference.zh-CN.md#models) | 特征顺序、归一化、张量形状、网络层、输出与损失 |
| [5. 校准与报价](technical-reference.zh-CN.md#quoting) | 波动率校准、信号组合、报价中心、宽度与 tick 取整 |
| [6. 执行与库存](technical-reference.zh-CN.md#execution) | 订单生命周期、库存过滤、并发、时间控制与关停 |
| [7. 配置与排错](technical-reference.zh-CN.md#operations) | 环境变量、依赖、日志和运行时文件 |
| [8. 训练与模型文件](technical-reference.zh-CN.md#training) | 训练流程、checkpoint 元数据、保存结果和缺失的数据集 |
| [9. 问题与安全](technical-reference.zh-CN.md#issues) | 实现问题、条件性风险、账户数据处理与安全 |

模型入口：[A：收益率分布](technical-reference.zh-CN.md#model-a) · [B：短期分位数](technical-reference.zh-CN.md#model-b) · [C：盘口中价与半价差](technical-reference.zh-CN.md#model-c)

---

## 阅读路线

| 阅读目标 | 建议顺序 |
|---|---|
| 理解策略 | [合约与概率](technical-reference.zh-CN.md#contracts) → [三个模型](technical-reference.zh-CN.md#models) → [报价构造](technical-reference.zh-CN.md#quoting) |
| 理解实现 | [运行流程与共享状态](technical-reference.zh-CN.md#runtime) → [订单执行](technical-reference.zh-CN.md#execution) → [配置](technical-reference.zh-CN.md#operations) |
| 研究训练 | [模型架构与损失](technical-reference.zh-CN.md#models) → [模型文件与数据现状](technical-reference.zh-CN.md#training) |

> [!IMPORTANT]
> **已归档，不再维护。** **本仓库中的全部代码和文档均由 AI 生成。** 代码通过**聊天界面使用 GPT-5 生成**；文档使用 **Opus 和 GPT-6** 生成。
>
> **代码与文档的正确性均未经人工审阅。** 作者从未审阅过用于实盘交易的代码。文档可能与实现存在冲突；了解系统的实际行为时，请以源码为准。
>
> 该策略曾在 BTC 上盈利，并于 **2025 年 12 月下旬**停止盈利。**其利润与头部账户相比，仅占极小的一部分。**
>
> 实盘系统曾持续修改。本仓库保留了大体相同的实现，但代码和模型参数可能与实际交易时使用的版本有所不同，不一定对应某个具体的实盘版本。

> [!WARNING]
> **请勿将本系统连接到有资金的账户。** 系统会记录**明文 API 认证信息**，部分风控检查会直接终止进程，**却不撤销仍在盘口上的订单**。
>
> **使用本仓库的风险由使用者自行承担。** 本仓库仅供研究与算法交流，不提供任何保证或投资建议。**在适用法律允许的范围内，作者不对使用本仓库造成的交易损失或其他损害承担责任。** 具体条款以 [LICENSE](../LICENSE) 为准。

[问题章节](technical-reference.zh-CN.md#issues)说明了各条阅读路线涉及的实现风险。

---

## 阅读约定

特征下标从零开始。张量形状中的 `N` 表示 batch size，`L` 表示序列长度，`D` 表示特征维度。代码标识符和配置名称保留源码写法。
