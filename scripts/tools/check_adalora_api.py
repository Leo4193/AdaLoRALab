import os
import sys
import inspect

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import hf_compat

from transformers import AutoModelForSequenceClassification
from peft import AdaLoraConfig, TaskType, get_peft_model


def run():
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
    )

    cfg = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=12,
        target_r=2,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "query",
            "value",
        ],
        total_step=100,
        tinit=10,
        tfinal=80,
        deltaT=10,
        orth_reg_weight=0.1,
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, cfg)

    print("模型类型：", type(model))
    print("base_model 类型：", type(getattr(model, "base_model", None)))
    print("base_model.model 类型：", type(getattr(getattr(model, "base_model", None), "model", None)))

    candidates = {
        "model": model,
        "model.base_model": getattr(model, "base_model", None),
        "model.base_model.model": getattr(getattr(model, "base_model", None), "model", None),
    }

    for name, obj in candidates.items():
        print("=" * 80)
        print(name, type(obj))

        if obj is None:
            continue

        print("has update_and_allocate:", hasattr(obj, "update_and_allocate"))
        print("has rankallocator:", hasattr(obj, "rankallocator"))
        print("has rankallocator 属性:", "rankallocator" in dir(obj))
        print("包含 allocate/update/rank 的方法：")
        for item in dir(obj):
            low = item.lower()
            if "allocate" in low or "rank" in low or "mask" in low:
                print("  ", item)

        if hasattr(obj, "update_and_allocate"):
            print("update_and_allocate 签名：")
            print(inspect.signature(obj.update_and_allocate))

    print("=" * 80)
    print("包含 lora_E 的参数：")
    count = 0
    for name, p in model.named_parameters():
        if "lora_E" in name:
            print(name, tuple(p.shape), "requires_grad=", p.requires_grad)
            count += 1

    print("lora_E 数量：", count)


if __name__ == "__main__":
    run()