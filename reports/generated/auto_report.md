# AdaRankLab Pro 实验报告

## 1. 项目定位

AdaRankLab Pro 是一个面向参数高效微调的 AdaLoRA 自动化评估、动态秩诊断与策略推荐系统。系统围绕 LoRA 固定秩分配局限、AdaLoRA 动态秩剪枝、参数重要性评分和预算调度策略展开，实现了从训练、评估、诊断到报告生成的完整流程。

## 2. 实验概况

- 实验总数：22
- 数据集数量：3
- 方法数量：5
- 图表数量：39

## 3. 主要实验类型

- SST-2 多 rank 对比实验
- SST-2 / MRPC / RTE 多任务泛化实验
- Attention / FFN / Full 模块消融实验
- 严格 AdaLoRA 动态秩剪枝实验
- 参数×梯度重要性评分实验
- AdaRank Score 多维评分实验
- strict / fast / slow 预算调度对比实验

## 4. AdaRank Score Top 5

| 排名 | 实验 | AdaRank Score | Accuracy | F1 |
|---:|---|---:|---:|---:|
| 1 | sst2_lora_r2_e3 | 64.5042 | 0.873853 | 0.879121 |
| 2 | sst2_attn_lora_r4_e3 | 62.6544 | 0.885321 | 0.889381 |
| 3 | sst2_lora_r4_e3 | 62.6055 | 0.885321 | 0.889381 |
| 4 | sst2_adalora_strict_rank | 62.3499 | 0.866972 | 0.872247 |
| 5 | mrpc_lora_r4_e3 | 61.6369 | 0.828431 | 0.880952 |

## 5. 严格 AdaLoRA Rank 诊断结论

严格手写训练循环下，AdaLoRA 最终剪枝比例为 44.44%，effective rank 分布范围为 3 到 10。query 平均 rank 为 5.75，value 平均 rank 为 7.58，说明 value 子模块获得更多有效秩预算。

## 6. 重要性评分结论

重要性评分实验显示，query 平均重要性为 1.080656e-06，value 平均重要性为 2.395986e-03。重要性与最终 rank 的相关系数约为 0.4642，说明重要性评分与最终秩保留存在正相关。

## 7. 预算调度结论

预算调度对比实验中，slow 策略取得最佳性能，Accuracy=0.878440，F1=0.883516。

## 8. 推荐图表

- `results/figs/comparison/ablation/module_ablation_accuracy.png`：模块消融实验
- `results/figs/comparison/ablation/module_ablation_f1.png`：模块消融实验
- `results/figs/comparison/ablation/module_param_accuracy.png`：模块消融实验
- `results/figs/comparison/ablation/module_train_time.png`：模块消融实验
- `results/figs/budget/budget_accuracy_f1.png`：预算调度策略分析
- `results/figs/budget/budget_query_value_rank.png`：预算调度策略分析
- `results/figs/budget/budget_rank_distribution_fast.png`：预算调度策略分析
- `results/figs/budget/budget_rank_distribution_slow.png`：预算调度策略分析
- `results/figs/budget/budget_rank_distribution_strict.png`：严格 AdaLoRA 动态秩诊断
- `results/figs/budget/budget_zero_count_curve.png`：预算调度策略分析
- `results/figs/diagnostics/importance/importance_change_curve.png`：重要性评分分析
- `results/figs/diagnostics/importance/importance_heatmap.png`：重要性评分分析
- `results/figs/diagnostics/importance/importance_vs_rank.png`：重要性评分分析
- `results/figs/diagnostics/importance/layer_importance.png`：重要性评分分析
- `results/figs/diagnostics/importance/query_value_importance.png`：重要性评分分析
- `results/figs/diagnostics/importance/top12_importance_modules.png`：模块消融实验
- `results/figs/comparison/multitask/multitask_accuracy.png`：多任务泛化实验
- `results/figs/comparison/multitask/multitask_accuracy_gap.png`：多任务泛化实验
- `results/figs/comparison/multitask/multitask_f1.png`：多任务泛化实验
- `results/figs/comparison/multitask/multitask_train_time.png`：多任务泛化实验

## 9. 报告刷新说明

本报告随 metrics、rank、importance 和图表文件刷新，内容用于汇总当前实验状态。