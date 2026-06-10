# AdaRankLab系统手册

## 1. 项目概览

AdaRankLab 面向参数高效微调实验，围绕 LoRA、AdaLoRA、动态 rank 诊断、参数重要性评分、预算调度对比和 AdaRank Score 展开。项目提供训练、可视化、评分、报告汇总和策略推荐入口，当前文件已按实验类型和产物类型重新归档。

项目根目录：

```text
D:\Github\AdaLoRALab
```

主要环境：

```text
操作系统：Windows
Python：3.11.9
基础模型：bert-base-uncased
数据集：GLUE / SST-2、MRPC、RTE
框架：PyTorch、Transformers、PEFT、Datasets、Scikit-learn、Pandas、Matplotlib
```

## 2. 最新目录结构

```text
AdaLoRALab/
├── configs/
│   ├── baselines/       # SST-2 LoRA / AdaLoRA 多 rank 与小样本配置
│   ├── multitask/       # MRPC、RTE 多任务配置
│   ├── ablation/        # Attention、FFN、Full 模块消融配置
│   ├── diagnostics/     # 严格 rank 与重要性诊断配置
│   └── budget/          # fast / slow 预算调度配置
├── scripts/
│   ├── run_tool.py      # 统一入口
│   ├── train/           # 训练脚本
│   ├── visualize/       # 图表脚本
│   ├── report/          # 评分、报告、推荐脚本
│   └── tools/           # 检查工具
├── src/
│   └── hf_compat.py     # HuggingFace / Transformers 兼容处理
├── results/
│   ├── metrics/         # 当前指标表与历史快照
│   ├── rank/            # rank 诊断、预算日志与归档
│   ├── importance/      # 重要性日志
│   ├── score/           # AdaRank Score 表
│   └── figs/            # 图表产物
├── reports/
│   ├── summary/         # 核心结论和评分摘要
│   ├── tables/          # 实验总表、图表清单、文件清单
│   └── generated/       # 实验报告和策略推荐
└── 手册.md
```

不建议提交或展示的目录：

```text
.venv/
.hf_cache/
outputs/
__pycache__/
```

## 3. 配置文件分类

`configs/baselines/`

```text
sst2_lora2_e3.yaml
sst2_lora4.yaml
sst2_lora4_e3.yaml
sst2_lora8_e3.yaml
sst2_lora16_e3.yaml
sst2_adalora2_e3.yaml
sst2_adalora4.yaml
sst2_adalora4_e3.yaml
sst2_adalora8_e3.yaml
sst2_adalora16_e3.yaml
```

`configs/multitask/`

```text
mrpc_lora4_e3.yaml
mrpc_adalora4_e3.yaml
rte_lora4_e3.yaml
rte_adalora4_e3.yaml
```

`configs/ablation/`

```text
sst2_attn_lora4_e3.yaml
sst2_ffn_lora4_e3.yaml
sst2_full_lora4_e3.yaml
sst2_full_adalora4_rank_e3.yaml
```

`configs/diagnostics/`

```text
sst2_adalora_strict_rank.yaml
sst2_adalora_importance.yaml
```

`configs/budget/`

```text
sst2_adalora_budget_fast.yaml
sst2_adalora_budget_slow.yaml
```

`run_tool.py` 支持用配置文件名自动查找归档位置，例如：

```powershell
python scripts\run_tool.py --mode train --script train02 --config sst2_lora4_e3
```

也可以使用完整路径：

```powershell
python scripts\run_tool.py --mode train --script train05 --config configs\diagnostics\sst2_adalora_strict_rank.yaml
```

## 4. 脚本分类

训练脚本位于 `scripts/train/`：

```text
train.py       初始 LoRA / AdaLoRA 训练入口
train01.py     初始训练副本
train02.py     多 rank、多任务正式训练
train03.py     Trainer 回调式 rank 诊断
train04.py     模块消融训练
train05.py     严格 AdaLoRA 动态 rank 训练
train06.py     重要性评分训练
```

图表脚本位于 `scripts/visualize/`：

```text
show01.py      SST-2 多 rank 对比
show02.py      多任务泛化对比
show03.py      模块消融对比
show04.py      Rank 强度分析
show05.py      严格 AdaLoRA rank 诊断
show06.py      参数重要性分析
show07.py      预算调度对比
```

报告脚本位于 `scripts/report/`：

```text
score01.py     AdaRank Score 评分
report01.py    实验报告汇总
advisor01.py   微调策略推荐
```

工具脚本位于 `scripts/tools/`：

```text
check_adalora_api.py
```

## 5. 统一入口

统一入口仍位于：

```text
scripts/run_tool.py
```

查看状态：

```powershell
python scripts\run_tool.py --mode status
```

训练实验：

```powershell
python scripts\run_tool.py --mode train --script train02 --config sst2_lora4_e3
python scripts\run_tool.py --mode train --script train05 --config sst2_adalora_strict_rank
python scripts\run_tool.py --mode train --script train06 --config sst2_adalora_importance
```

生成图表：

```powershell
python scripts\run_tool.py --mode show --name sst2_rank
python scripts\run_tool.py --mode show --name multitask
python scripts\run_tool.py --mode show --name ablation
python scripts\run_tool.py --mode show --name strict_rank
python scripts\run_tool.py --mode show --name importance
python scripts\run_tool.py --mode show --name budget
```

评分、报告和推荐：

```powershell
python scripts\run_tool.py --mode score
python scripts\run_tool.py --mode report
python scripts\run_tool.py --mode advisor
```

刷新全部图表、评分、报告和推荐：

```powershell
python scripts\run_tool.py --mode all_report
```

## 6. 结果目录

指标：

```text
results/metrics/metrics.csv
results/metrics/archive/
```

Rank 日志：

```text
results/rank/diagnostics/sst2_adalora_strict_rank_rank.csv
results/rank/diagnostics/sst2_adalora_importance_rank.csv
results/rank/diagnostics/sst2_full_adalora_r4_rank_e3_rank.csv
results/rank/budget/sst2_adalora_budget_fast_rank.csv
results/rank/budget/sst2_adalora_budget_slow_rank.csv
results/rank/archive/strict_rank_backup.csv
```

重要性日志：

```text
results/importance/diagnostics/sst2_adalora_importance.csv
```

AdaRank Score：

```text
results/score/adarank_score.csv
results/figs/score/adarank_score_bar.png
results/figs/score/adarank_dimension_bar.png
results/figs/score/adarank_score_radar.png
```

图表：

```text
results/figs/comparison/sst2_rank/
results/figs/comparison/multitask/
results/figs/comparison/ablation/
results/figs/diagnostics/rank_strength/
results/figs/diagnostics/strict_rank/
results/figs/diagnostics/importance/
results/figs/budget/
results/figs/score/
```

## 7. 报告目录

摘要：

```text
reports/summary/核心结论摘要.md
reports/summary/adarank_score_summary.md
```

表格和清单：

```text
reports/tables/实验总表.csv
reports/tables/图表清单.csv
reports/tables/图表清单.md
reports/tables/project_inventory.md
```

报告和推荐：

```text
reports/generated/auto_report.md
reports/generated/advisor_suggestions.md
```

## 8. 主要实验线

SST-2 多 rank 对比：

```text
LoRA：r=2、4、8、16
AdaLoRA：r=2、4、8、16
图表目录：results/figs/comparison/sst2_rank/
```

多任务泛化：

```text
任务：SST-2、MRPC、RTE
图表目录：results/figs/comparison/multitask/
```

模块消融：

```text
Attention-only
FFN-only
Attention+FFN
图表目录：results/figs/comparison/ablation/
```

严格 AdaLoRA 动态 rank：

```text
配置：configs/diagnostics/sst2_adalora_strict_rank.yaml
Rank 日志：results/rank/diagnostics/sst2_adalora_strict_rank_rank.csv
图表目录：results/figs/diagnostics/strict_rank/
```

重要性评分：

```text
配置：configs/diagnostics/sst2_adalora_importance.yaml
重要性日志：results/importance/diagnostics/sst2_adalora_importance.csv
图表目录：results/figs/diagnostics/importance/
```

预算调度：

```text
配置：configs/budget/sst2_adalora_budget_fast.yaml
配置：configs/budget/sst2_adalora_budget_slow.yaml
图表目录：results/figs/budget/
```

## 9. Demo 展示流程

```powershell
python scripts\run_tool.py --mode status
python scripts\run_tool.py --mode all_report
```

重点查看：

```text
results/figs/comparison/sst2_rank/sst2_acc_rank.png
results/figs/comparison/ablation/module_ablation_f1.png
results/figs/diagnostics/strict_rank/strict_rank_heatmap.png
results/figs/diagnostics/importance/importance_vs_rank.png
results/figs/budget/budget_accuracy_f1.png
results/figs/score/adarank_score_bar.png
reports/generated/auto_report.md
reports/generated/advisor_suggestions.md
```

## 10. 常见问题

如果 HuggingFace 下载较慢：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

如果 `Trainer.__init__()` 的 `tokenizer` 参数报错，训练脚本会按当前 Transformers 接口选择 `processing_class` 或 `tokenizer`。

如果需要直接运行归档后的脚本，请从项目根目录执行，例如：

```powershell
python scripts\visualize\show05.py
python scripts\report\report01.py
```

推荐优先通过 `scripts/run_tool.py` 运行，入口会处理脚本别名和配置文件查找。
