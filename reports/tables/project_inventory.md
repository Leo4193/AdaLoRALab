# 项目文件清单

## configs

- `configs/ablation/sst2_attn_lora4_e3.yaml`
- `configs/ablation/sst2_ffn_lora4_e3.yaml`
- `configs/ablation/sst2_full_adalora4_rank_e3.yaml`
- `configs/ablation/sst2_full_lora4_e3.yaml`
- `configs/baselines/sst2_adalora16_e3.yaml`
- `configs/baselines/sst2_adalora2_e3.yaml`
- `configs/baselines/sst2_adalora4.yaml`
- `configs/baselines/sst2_adalora4_e3.yaml`
- `configs/baselines/sst2_adalora8_e3.yaml`
- `configs/baselines/sst2_lora16_e3.yaml`
- `configs/baselines/sst2_lora2_e3.yaml`
- `configs/baselines/sst2_lora4.yaml`
- `configs/baselines/sst2_lora4_e3.yaml`
- `configs/baselines/sst2_lora8_e3.yaml`
- `configs/budget/sst2_adalora_budget_fast.yaml`
- `configs/budget/sst2_adalora_budget_slow.yaml`
- `configs/diagnostics/sst2_adalora_importance.yaml`
- `configs/diagnostics/sst2_adalora_strict_rank.yaml`
- `configs/multitask/mrpc_adalora4_e3.yaml`
- `configs/multitask/mrpc_lora4_e3.yaml`
- `configs/multitask/rte_adalora4_e3.yaml`
- `configs/multitask/rte_lora4_e3.yaml`

## scripts

- `scripts/report/advisor01.py`
- `scripts/report/report01.py`
- `scripts/report/score01.py`
- `scripts/run_tool.py`
- `scripts/tools/check_adalora_api.py`
- `scripts/train/train.py`
- `scripts/train/train01.py`
- `scripts/train/train02.py`
- `scripts/train/train03.py`
- `scripts/train/train04.py`
- `scripts/train/train05.py`
- `scripts/train/train06.py`
- `scripts/visualize/show01.py`
- `scripts/visualize/show02.py`
- `scripts/visualize/show03.py`
- `scripts/visualize/show04.py`
- `scripts/visualize/show05.py`
- `scripts/visualize/show06.py`
- `scripts/visualize/show07.py`

## results/metrics

- `results/metrics/archive/metrics_after_ablation.csv`
- `results/metrics/archive/metrics_after_multitask.csv`
- `results/metrics/archive/metrics_after_sst2_rank.csv`
- `results/metrics/archive/metrics_after_strict_adalora.csv`
- `results/metrics/archive/metrics_sst2_rank.csv`
- `results/metrics/metrics.csv`

## results/rank

- `results/rank/archive/strict_rank_backup.csv`
- `results/rank/budget/sst2_adalora_budget_fast_rank.csv`
- `results/rank/budget/sst2_adalora_budget_slow_rank.csv`
- `results/rank/diagnostics/sst2_adalora_importance_rank.csv`
- `results/rank/diagnostics/sst2_adalora_strict_rank_rank.csv`
- `results/rank/diagnostics/sst2_full_adalora_r4_rank_e3_rank.csv`

## results/importance

- `results/importance/diagnostics/sst2_adalora_importance.csv`

## results/score

- `results/score/adarank_score.csv`

## results/figs

- `results/figs/budget/budget_accuracy_f1.png`
- `results/figs/budget/budget_query_value_rank.png`
- `results/figs/budget/budget_rank_distribution_fast.png`
- `results/figs/budget/budget_rank_distribution_slow.png`
- `results/figs/budget/budget_rank_distribution_strict.png`
- `results/figs/budget/budget_result_table.csv`
- `results/figs/budget/budget_summary.md`
- `results/figs/budget/budget_zero_count_curve.png`
- `results/figs/comparison/ablation/module_ablation_accuracy.png`
- `results/figs/comparison/ablation/module_ablation_f1.png`
- `results/figs/comparison/ablation/module_ablation_result.csv`
- `results/figs/comparison/ablation/module_param_accuracy.png`
- `results/figs/comparison/ablation/module_train_time.png`
- `results/figs/comparison/multitask/multitask_accuracy.png`
- `results/figs/comparison/multitask/multitask_accuracy_gap.png`
- `results/figs/comparison/multitask/multitask_f1.png`
- `results/figs/comparison/multitask/multitask_result.csv`
- `results/figs/comparison/multitask/multitask_train_time.png`
- `results/figs/comparison/sst2_rank/sst2_acc_rank.png`
- `results/figs/comparison/sst2_rank/sst2_eval_loss_curve.png`
- `results/figs/comparison/sst2_rank/sst2_f1_rank.png`
- `results/figs/comparison/sst2_rank/sst2_pareto_acc.png`
- `results/figs/comparison/sst2_rank/sst2_rank_result.csv`
- `results/figs/comparison/sst2_rank/sst2_train_time.png`
- `results/figs/diagnostics/importance/final_importance_table.csv`
- `results/figs/diagnostics/importance/importance_change_curve.png`
- `results/figs/diagnostics/importance/importance_heatmap.png`
- `results/figs/diagnostics/importance/importance_rank_merged.csv`
- `results/figs/diagnostics/importance/importance_summary.md`
- `results/figs/diagnostics/importance/importance_vs_rank.png`
- `results/figs/diagnostics/importance/layer_importance.png`
- `results/figs/diagnostics/importance/query_value_importance.png`
- `results/figs/diagnostics/importance/top12_importance_modules.csv`
- `results/figs/diagnostics/importance/top12_importance_modules.png`
- `results/figs/diagnostics/rank_strength/final_rank_table.csv`
- `results/figs/diagnostics/rank_strength/layer_group_rank.png`
- `results/figs/diagnostics/rank_strength/layer_total_rank.png`
- `results/figs/diagnostics/rank_strength/module_rank_bar.png`
- `results/figs/diagnostics/rank_strength/rank_change_curve.png`
- `results/figs/diagnostics/rank_strength/rank_heatmap.png`
- `results/figs/diagnostics/strict_rank/strict_final_rank_table.csv`
- `results/figs/diagnostics/strict_rank/strict_layer_rank.png`
- `results/figs/diagnostics/strict_rank/strict_query_value_rank.png`
- `results/figs/diagnostics/strict_rank/strict_rank_change_curve.png`
- `results/figs/diagnostics/strict_rank/strict_rank_distribution.png`
- `results/figs/diagnostics/strict_rank/strict_rank_heatmap.png`
- `results/figs/diagnostics/strict_rank/strict_zero_count_curve.png`
- `results/figs/score/adarank_dimension_bar.png`
- `results/figs/score/adarank_score_bar.png`
- `results/figs/score/adarank_score_radar.png`

## reports

- `reports/generated/advisor_suggestions.md`
- `reports/generated/auto_report.md`
- `reports/summary/abstract.md`
- `reports/summary/adarank_score_summary.md`
- `reports/summary/核心结论摘要.md`
- `reports/tables/project_inventory.md`
- `reports/tables/图表清单.csv`
- `reports/tables/图表清单.md`
- `reports/tables/实验总表.csv`
