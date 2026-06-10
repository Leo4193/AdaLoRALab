# 核心结论摘要

## 1. 实验规模

- 共记录实验数量：22
- 涉及数据集：mrpc, rte, sst2
- 涉及方法：adalora, adalora_budget_fast, adalora_importance, adalora_strict, lora

## 2. 各数据集最佳结果

| 数据集 | 指标 | 最佳实验 | 数值 |
|---|---|---|---:|
| mrpc | accuracy | mrpc_lora_r4_e3 | 0.828431 |
| mrpc | f1 | mrpc_lora_r4_e3 | 0.880952 |
| rte | accuracy | rte_lora_r4_e3 | 0.570397 |
| rte | f1 | rte_adalora_r4_e3 | 0.642157 |
| sst2 | accuracy | sst2_lora_r16_e3 | 0.907110 |
| sst2 | f1 | sst2_lora_r16_e3 | 0.909295 |

## 3. AdaRank Score 综合评价

- 综合评分最高实验：sst2_lora_r2_e3，AdaRank Score=64.5042。
- AdaRank Score 综合考虑任务性能、参数效率、训练效率、秩压缩、秩差异性和可解释性，不等同于单一 Accuracy 排名。

## 4. 严格 AdaLoRA 动态秩剪枝

- 最终 step：3750
- 诊断模块数：24
- 初始低秩方向数：288
- 最终保留方向数：160
- 置零方向数：128
- 剪枝比例：44.44%
- effective rank 范围：3 - 10
- query 平均 rank：5.75
- value 平均 rank：7.58

## 5. 重要性评分分析

- query 平均重要性：1.080656e-06
- value 平均重要性：2.395986e-03
- 重要性与最终 effective rank 的相关系数：0.4642
- Top 重要模块主要集中在 value 子模块，说明 value 对当前任务适配贡献更大。

## 6. 预算调度策略分析

- 最佳预算策略：slow，Accuracy=0.878440，F1=0.883516。
- 实验表明，较长预算调度阶段能够改善 AdaLoRA 的最终性能。

## 7. 总体结论

AdaRankLab Pro 已从单纯的 LoRA/AdaLoRA 复现实验扩展为参数高效微调评估、动态秩诊断、重要性分析、预算调度比较和策略推荐的完整工具型系统。