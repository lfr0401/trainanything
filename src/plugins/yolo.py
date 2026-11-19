"""YOLO 系列任务插件（detect / segment / classify / pose / obb）。

家族可用性依据 ultralytics 官方发布的预训练权重：
- YOLO12 官方仅发布 detect 权重，其余任务不含 YOLO12。
- YOLOv10 无 seg/cls/pose/obb；YOLOv9 只有 detect(c/e) 与 seg(c/e)。
"""

from .base import TaskPlugin, register

# 尺寸档位
S_FULL = (("Nano", "n"), ("Small", "s"), ("Medium", "m"), ("Large", "l"), ("ExtraLarge", "x"))
S_V10 = (("Nano", "n"), ("Small", "s"), ("Medium", "m"),
         ("Balanced", "b"), ("Large", "l"), ("ExtraLarge", "x"))
S_V9 = (("Compact", "c"), ("Enhanced", "e"))


def _f(label, prefix, sizes=S_FULL):
    """家族元组：(显示名, 权重前缀, 尺寸档位)"""
    return (label, prefix, sizes)


F_ALL = (
    _f("YOLO26", "yolo26"), _f("YOLO12", "yolo12"), _f("YOLO11", "yolo11"),
    _f("YOLOv10", "yolov10", S_V10), _f("YOLOv9", "yolov9", S_V9), _f("YOLOv8", "yolov8"),
)
F_SEG = (
    _f("YOLO26", "yolo26"), _f("YOLO11", "yolo11"),
    _f("YOLOv9", "yolov9", S_V9), _f("YOLOv8", "yolov8"),
)
F_CORE = (_f("YOLO26", "yolo26"), _f("YOLO11", "yolo11"), _f("YOLOv8", "yolov8"))


def register_yolo_plugins() -> None:
    register(TaskPlugin(
        key="detect", model_suffix="", families=F_ALL,
        dataset_kind="file", supports_track=True,
    ))
    register(TaskPlugin(
        key="segment", model_suffix="-seg", families=F_SEG,
        dataset_kind="file", supports_track=True,
    ))
    register(TaskPlugin(
        key="classify", model_suffix="-cls", families=F_CORE,
        dataset_kind="folder", supports_track=False,
    ))
    register(TaskPlugin(
        key="pose", model_suffix="-pose", families=F_CORE,
        dataset_kind="file", supports_track=True,
    ))
    register(TaskPlugin(
        key="obb", model_suffix="-obb", families=F_CORE,
        dataset_kind="file", supports_track=True,
    ))
