"""YOLO 算法插件。

一个算法 = 一个可执行引擎（src.train / src.detect）+ 它在多个任务下的能力。
任务 detect / segment / classify / pose / obb 都是 YOLO 家族提供的。

家族可用性依据 ultralytics 官方发布的预训练权重：
- YOLO12 官方仅发布 detect 权重，其余任务不含 YOLO12。
- YOLOv10 无 seg/cls/pose/obb；YOLOv9 只有 detect(c/e) 与 seg(c/e)。
"""

from .base import AlgorithmPlugin, Capability, register_algorithm

# 尺寸档位
S_FULL = (("Nano", "n"), ("Small", "s"), ("Medium", "m"), ("Large", "l"), ("ExtraLarge", "x"))
S_V10 = (("Nano", "n"), ("Small", "s"), ("Medium", "m"),
         ("Balanced", "b"), ("Large", "l"), ("ExtraLarge", "x"))
S_V9 = (("Compact", "c"), ("Enhanced", "e"))


def _f(label, prefix, sizes=S_FULL):
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


def register_yolo_algorithm() -> None:
    register_algorithm(AlgorithmPlugin(
        key="yolo",
        label_key="algo_yolo",
        train_module="src.train",
        detect_module="src.detect",
        detect_func="detect_images",
        capabilities=(
            Capability(task="detect", families=F_ALL, dataset_kind="file", supports_track=True),
            Capability(task="segment", families=F_SEG, model_suffix="-seg",
                       dataset_kind="file", supports_track=True),
            Capability(task="classify", families=F_CORE, model_suffix="-cls",
                       dataset_kind="folder", supports_track=False),
            Capability(task="pose", families=F_CORE, model_suffix="-pose",
                       dataset_kind="file", supports_track=True),
            Capability(task="obb", families=F_CORE, model_suffix="-obb",
                       dataset_kind="file", supports_track=True),
        ),
    ))
