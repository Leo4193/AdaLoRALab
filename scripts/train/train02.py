import os
import sys

# 把项目根目录加入 Python 路径，方便导入 src 里的兼容模块
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 必须放在 transformers / datasets / peft 导入之前
from src import hf_compat

import csv
import time
import yaml
import argparse
import inspect
import random
import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from peft import (
    LoraConfig,
    AdaLoraConfig,
    TaskType,
    get_peft_model,
)


def set_seed(seed):
    # 固定随机种子，尽量保证实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path):
    # 读取 yaml 配置文件
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_params(model):
    # 统计总参数和可训练参数
    total = 0
    trainable = 0

    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num

    ratio = trainable / total * 100
    return total, trainable, ratio


def load_data(cfg, tokenizer):
    print("正在加载数据集...")

    raw = load_dataset(cfg["dataset_name"], cfg["task_name"])

    train_data = raw["train"]
    eval_data = raw["validation"]

    # 可以通过配置文件控制是否只取部分样本
    if cfg.get("max_train_samples") is not None:
        n = min(int(cfg["max_train_samples"]), len(train_data))
        train_data = train_data.select(range(n))

    if cfg.get("max_eval_samples") is not None:
        n = min(int(cfg["max_eval_samples"]), len(eval_data))
        eval_data = eval_data.select(range(n))

    task_name = cfg["task_name"]

    def tokenize(batch):
        if task_name == "sst2":
            return tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=int(cfg["max_length"]),
            )
        elif task_name in ["mrpc", "rte"]:
            return tokenizer(
                batch["sentence1"],
                batch["sentence2"],
                truncation=True,
                max_length=int(cfg["max_length"]),
            )
        else:
            raise ValueError(f"暂不支持这个任务：{task_name}")

    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)

    print(f"训练样本数：{len(train_data)}")
    print(f"验证样本数：{len(eval_data)}")

    return train_data, eval_data


def build_model(cfg):
    print("正在加载模型...")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
    )

    method = cfg["method"].lower()

    if method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(cfg["rank"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            target_modules=cfg["target_modules"],
            bias="none",
            modules_to_save=["classifier"],
        )

    elif method == "adalora":
        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            init_r=int(cfg.get("init_r", 8)),
            target_r=int(cfg.get("target_r", cfg["rank"])),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            target_modules=cfg["target_modules"],
            bias="none",
            modules_to_save=["classifier"],

            # AdaLoRA 动态秩调度参数
            total_step=int(cfg.get("total_step", 1000)),
            tinit=int(cfg.get("tinit", 100)),
            tfinal=int(cfg.get("tfinal", 800)),
            deltaT=int(cfg.get("deltaT", 50)),
            orth_reg_weight=float(cfg.get("orth_reg_weight", 0.1)),
        )

    else:
        raise ValueError(f"暂不支持这个方法：{method}")

    model = get_peft_model(model, peft_config)

    print("可训练参数如下：")
    model.print_trainable_parameters()

    return model


def make_train_args(cfg):
    # transformers 5.x 和旧版 transformers 参数名略有差异，因此用反射自动兼容
    args_dict = {
        "output_dir": cfg["output_dir"],
        "learning_rate": float(cfg["learning_rate"]),
        "per_device_train_batch_size": int(cfg["train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["eval_batch_size"]),
        "num_train_epochs": float(cfg["num_train_epochs"]),
        "weight_decay": float(cfg["weight_decay"]),
        "logging_steps": 50,
        "save_strategy": "epoch",
        "report_to": "none",
        "seed": int(cfg["seed"]),
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": 0,
    }

    sig = inspect.signature(TrainingArguments.__init__)

    # 兼容 evaluation_strategy / eval_strategy
    if "eval_strategy" in sig.parameters:
        args_dict["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig.parameters:
        args_dict["evaluation_strategy"] = "epoch"

    # 兼容 logging_dir / tensorboard_logging_dir
    if "tensorboard_logging_dir" in sig.parameters:
        args_dict["tensorboard_logging_dir"] = cfg["logging_dir"]
    elif "logging_dir" in sig.parameters:
        args_dict["logging_dir"] = cfg["logging_dir"]

    return TrainingArguments(**args_dict)


class AdaLoraStepCallback(TrainerCallback):
    # AdaLoRA 训练过程中需要按 step 更新秩预算
    # 这里采用反射写法，兼容不同 PEFT 版本
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        step = int(state.global_step)

        candidates = [
            model,
            getattr(model, "base_model", None),
            getattr(getattr(model, "base_model", None), "model", None),
        ]

        for obj in candidates:
            if obj is not None and hasattr(obj, "update_and_allocate"):
                try:
                    obj.update_and_allocate(step)
                    break
                except TypeError:
                    try:
                        obj.update_and_allocate(global_step=step)
                        break
                    except Exception:
                        pass
                except Exception:
                    pass

        return control


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
    }


def save_metrics(cfg, total_params, trainable_params, trainable_ratio, eval_result, train_time):
    os.makedirs(os.path.dirname(cfg["metric_file"]), exist_ok=True)

    file_exists = os.path.exists(cfg["metric_file"])

    row = {
        "exp_name": cfg["exp_name"],
        "dataset": cfg["task_name"],
        "model": cfg["model_name"],
        "method": cfg["method"],
        "rank": cfg["rank"],
        "train_samples": cfg.get("max_train_samples"),
        "eval_samples": cfg.get("max_eval_samples"),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio_percent": round(trainable_ratio, 4),
        "eval_loss": round(float(eval_result.get("eval_loss", 0)), 6),
        "accuracy": round(float(eval_result.get("eval_accuracy", 0)), 6),
        "f1": round(float(eval_result.get("eval_f1", 0)), 6),
        "train_time_sec": round(float(train_time), 2),
        "seed": cfg["seed"],
    }

    with open(cfg["metric_file"], "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print("指标已保存到：", cfg["metric_file"])
    print("本次实验结果：")
    for k, v in row.items():
        print(f"{k}: {v}")


def save_log_history(cfg, trainer):
    # 保存 Trainer 的日志历史，后面用于画 loss 曲线
    log_path = os.path.join("results", "logs", f"{cfg['exp_name']}_history.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    rows = trainer.state.log_history
    if not rows:
        print("没有可保存的训练日志。")
        return

    keys = sorted(set().union(*(row.keys() for row in rows)))

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("训练日志已保存到：", log_path)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    cfg = read_config(args.config)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["logging_dir"], exist_ok=True)

    set_seed(int(cfg["seed"]))

    print("=" * 60)
    print("实验名称：", cfg["exp_name"])
    print("微调方法：", cfg["method"])
    print("数据集：", cfg["task_name"])
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_data, eval_data = load_data(cfg, tokenizer)
    model = build_model(cfg)

    total_params, trainable_params, trainable_ratio = count_params(model)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_args = make_train_args(cfg)

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_data,
        "eval_dataset": eval_data,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    # transformers 5.x 不再使用 tokenizer 参数，改为 processing_class
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    # AdaLoRA 需要在训练过程中动态更新秩预算
    if cfg["method"].lower() == "adalora":
        trainer_kwargs["callbacks"] = [AdaLoraStepCallback()]

    trainer = Trainer(**trainer_kwargs)

    print("开始训练...")
    start = time.time()

    trainer.train()

    train_time = time.time() - start

    print("开始验证...")
    eval_result = trainer.evaluate()

    print("保存模型...")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    save_metrics(
        cfg,
        total_params,
        trainable_params,
        trainable_ratio,
        eval_result,
        train_time,
    )

    save_log_history(cfg, trainer)

    print("实验完成。")


if __name__ == "__main__":
    run()