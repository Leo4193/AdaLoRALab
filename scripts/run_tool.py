import argparse
import os
import subprocess
import sys
from datetime import datetime


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")


SCRIPT_ALIASES = {
    "train": "train/train.py",
    "train01": "train/train01.py",
    "train02": "train/train02.py",
    "train03": "train/train03.py",
    "train04": "train/train04.py",
    "train05": "train/train05.py",
    "train06": "train/train06.py",
    "show01": "visualize/show01.py",
    "show02": "visualize/show02.py",
    "show03": "visualize/show03.py",
    "show04": "visualize/show04.py",
    "show05": "visualize/show05.py",
    "show06": "visualize/show06.py",
    "show07": "visualize/show07.py",
    "score01": "report/score01.py",
    "report01": "report/report01.py",
    "advisor01": "report/advisor01.py",
    "check_adalora_api": "tools/check_adalora_api.py",
}


SHOW_ALIASES = {
    "sst2_rank": "show01",
    "multitask": "show02",
    "ablation": "show03",
    "rank_strength": "show04",
    "strict_rank": "show05",
    "importance": "show06",
    "budget": "show07",
}


STATUS_TARGETS = [
    "configs/baselines",
    "configs/multitask",
    "configs/ablation",
    "configs/diagnostics",
    "configs/budget",
    "results/metrics/metrics.csv",
    "results/metrics/archive",
    "results/rank/diagnostics/sst2_adalora_strict_rank_rank.csv",
    "results/rank/budget/sst2_adalora_budget_fast_rank.csv",
    "results/importance/diagnostics/sst2_adalora_importance.csv",
    "results/score/adarank_score.csv",
    "results/figs/comparison/sst2_rank",
    "results/figs/comparison/multitask",
    "results/figs/comparison/ablation",
    "results/figs/diagnostics/strict_rank",
    "results/figs/diagnostics/importance",
    "results/figs/budget",
    "results/figs/score",
    "reports/summary/核心结论摘要.md",
    "reports/summary/adarank_score_summary.md",
    "reports/tables/实验总表.csv",
    "reports/tables/图表清单.md",
    "reports/generated/auto_report.md",
    "reports/generated/advisor_suggestions.md",
]


def run_cmd(cmd):
    print("=" * 80)
    print("执行命令：")
    print(" ".join(cmd))
    print("=" * 80)

    start = datetime.now()
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    end = datetime.now()

    print("=" * 80)
    print("开始时间：", start.strftime("%Y-%m-%d %H:%M:%S"))
    print("结束时间：", end.strftime("%Y-%m-%d %H:%M:%S"))
    print("耗时：", end - start)
    print("=" * 80)

    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败，返回码：{result.returncode}")


def normalize_script_name(name):
    name = name.replace("\\", "/")
    if name.endswith(".py"):
        name = name[:-3]
    return name


def script_path(name):
    key = normalize_script_name(name)
    rel = SCRIPT_ALIASES.get(key, name.replace("\\", "/"))
    if not rel.endswith(".py"):
        rel = rel + ".py"

    candidates = [
        os.path.join(SCRIPT_DIR, rel),
        os.path.join(ROOT_DIR, rel),
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(f"找不到脚本：{name}")


def resolve_config(config):
    raw = config.replace("\\", "/")
    candidates = [
        os.path.join(ROOT_DIR, raw),
        os.path.join(CONFIG_DIR, raw),
    ]

    if not raw.endswith((".yaml", ".yml")):
        candidates.append(os.path.join(CONFIG_DIR, raw + ".yaml"))

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    name = os.path.basename(raw)
    if not name.endswith((".yaml", ".yml")):
        name = name + ".yaml"

    matches = []
    for dirpath, _, filenames in os.walk(CONFIG_DIR):
        for filename in filenames:
            if filename == name:
                matches.append(os.path.join(dirpath, filename))

    if len(matches) == 1:
        return os.path.abspath(matches[0])

    if len(matches) > 1:
        readable = [os.path.relpath(path, ROOT_DIR).replace("\\", "/") for path in matches]
        raise ValueError(f"配置文件名不唯一：{config}，匹配到：{readable}")

    raise FileNotFoundError(f"找不到配置文件：{config}")


def run_train(args):
    if args.script is None:
        raise ValueError("训练模式必须指定 --script，例如 --script train02")

    if args.config is None:
        raise ValueError("训练模式必须指定 --config，例如 --config sst2_lora4_e3")

    run_cmd([
        sys.executable,
        script_path(args.script),
        "--config",
        resolve_config(args.config),
    ])


def run_show(args):
    if args.name is None:
        print("可用 show 名称：")
        for name in SHOW_ALIASES:
            print(" -", name)
        raise ValueError("show 模式必须指定 --name")

    if args.name not in SHOW_ALIASES:
        raise ValueError(f"未知 show 名称：{args.name}，可用：{list(SHOW_ALIASES.keys())}")

    run_cmd([sys.executable, script_path(SHOW_ALIASES[args.name])])


def run_score():
    run_cmd([sys.executable, script_path("score01")])


def run_report():
    run_cmd([sys.executable, script_path("report01")])


def run_advisor():
    run_cmd([sys.executable, script_path("advisor01")])


def run_all_show():
    for name in ["sst2_rank", "multitask", "ablation", "strict_rank", "importance", "budget"]:
        print("\n\n")
        print("#" * 80)
        print("生成图表：", name)
        print("#" * 80)

        class Tmp:
            pass

        tmp = Tmp()
        tmp.name = name
        run_show(tmp)


def run_all_report():
    run_all_show()
    run_score()
    run_report()
    run_advisor()


def show_status():
    print("=" * 80)
    print("AdaRankLab Pro 项目状态检查")
    print("=" * 80)

    for item in STATUS_TARGETS:
        path = os.path.join(ROOT_DIR, item)
        mark = "OK" if os.path.exists(path) else "MISSING"
        print(f"[{mark}] {item}")

    print("=" * 80)


def show_help():
    print("""
AdaRankLab Pro 统一入口用法：

1. 训练实验：
   python scripts/run_tool.py --mode train --script train02 --config sst2_lora4_e3
   python scripts/run_tool.py --mode train --script train05 --config sst2_adalora_strict_rank

2. 生成单类图表：
   python scripts/run_tool.py --mode show --name sst2_rank
   python scripts/run_tool.py --mode show --name multitask
   python scripts/run_tool.py --mode show --name ablation
   python scripts/run_tool.py --mode show --name strict_rank
   python scripts/run_tool.py --mode show --name importance
   python scripts/run_tool.py --mode show --name budget

3. 生成评分、报告和推荐：
   python scripts/run_tool.py --mode score
   python scripts/run_tool.py --mode report
   python scripts/run_tool.py --mode advisor

4. 刷新全部图表、评分、报告和推荐：
   python scripts/run_tool.py --mode all_report

5. 检查项目状态：
   python scripts/run_tool.py --mode status
""")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="help",
        choices=[
            "train",
            "show",
            "score",
            "report",
            "advisor",
            "all_show",
            "all_report",
            "status",
            "help",
        ],
        help="运行模式",
    )
    parser.add_argument("--script", type=str, default=None, help="训练脚本名，例如 train02")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径或文件名")
    parser.add_argument("--name", type=str, default=None, help="show 图表名称")

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "show":
        run_show(args)
    elif args.mode == "score":
        run_score()
    elif args.mode == "report":
        run_report()
    elif args.mode == "advisor":
        run_advisor()
    elif args.mode == "all_show":
        run_all_show()
    elif args.mode == "all_report":
        run_all_report()
    elif args.mode == "status":
        show_status()
    else:
        show_help()


if __name__ == "__main__":
    run()
