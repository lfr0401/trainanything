"""插件注册表。

设计原则：
- 本包只放**元数据**（任务有哪些、每个任务可选什么模型、数据集是文件还是目录），
  纯 Python 数据，不 import torch / ultralytics / cv2，确保 GUI 启动零成本、界面不卡。
- 训练与推理的实际执行仍在 src/train.py、src/detect.py、src/camera.py 的子进程/子线程中。
- 新增任务 = 新建一个模块调用 register() + 在 __init__.py 导入一次。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskPlugin:
    """一个任务插件的声明。

    key:            任务标识，如 detect / segment，对应 gui_text 里的 task_<key> 文案键
    model_suffix:   预训练权重后缀，如 '' / '-seg' / '-cls'
    families:       支持的家族，每项 (家族显示名, 权重前缀, 尺寸档位)
                    尺寸档位为 ((显示名, 后缀), ...)，每个家族可不同
                    （例：yolov9 只有 c/e，其余是 n/s/m/l/x）
    dataset_kind:   'file' 选 data.yaml；'folder' 选数据集根目录（分类）
    supports_track: 该任务是否支持视频/摄像头目标跟踪
    """

    key: str
    model_suffix: str
    families: Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]
    dataset_kind: str = "file"
    supports_track: bool = False


_REGISTRY: Dict[str, TaskPlugin] = {}


def register(plugin: TaskPlugin) -> None:
    """注册一个任务插件；重复 key 直接覆盖，便于扩展。"""
    _REGISTRY[plugin.key] = plugin


def get_plugin(key: str) -> Optional[TaskPlugin]:
    return _REGISTRY.get(key)


def all_plugins() -> List[TaskPlugin]:
    """按注册顺序返回全部插件。"""
    return list(_REGISTRY.values())


def task_keys() -> List[str]:
    return [p.key for p in all_plugins()]


def model_options_by_task() -> Dict[str, List[Tuple[str, str]]]:
    """{任务: [(显示名, 权重名), ...]}"""
    result = {}
    for p in all_plugins():
        options = []
        for family_label, family_prefix, sizes in p.families:
            for size_label, size_suffix in sizes:
                options.append((
                    f"{family_label}-{size_label}{p.model_suffix}",
                    f"{family_prefix}{size_suffix}{p.model_suffix}",
                ))
        result[p.key] = options
    return result


def model_name_to_type_by_task(options: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Dict[str, str]]:
    return {task: dict(opts) for task, opts in options.items()}


def supports_track(task: str) -> bool:
    p = get_plugin(task)
    return bool(p and p.supports_track)


def dataset_kind(task: str) -> str:
    p = get_plugin(task)
    return p.dataset_kind if p else "file"
