import os
import huggingface_hub


def _str_to_bool(value):
    if value is None:
        return False
    return str(value).strip().upper() in {"1", "ON", "YES", "TRUE"}


def _fallback_is_offline_mode():
    return (
        _str_to_bool(os.environ.get("HF_HUB_OFFLINE"))
        or _str_to_bool(os.environ.get("TRANSFORMERS_OFFLINE"))
    )


# transformers 5.x 可能从 huggingface_hub 顶层导入 is_offline_mode
# 不同 hub 版本有时把它放在 utils 里，所以这里统一补到顶层
if not hasattr(huggingface_hub, "is_offline_mode"):
    try:
        from huggingface_hub.utils import is_offline_mode
        huggingface_hub.is_offline_mode = is_offline_mode
    except Exception:
        huggingface_hub.is_offline_mode = _fallback_is_offline_mode
