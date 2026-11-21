"""语义分割算法插件（semseg 引擎）。

与 YOLO 实例分割的区别：
- YOLO segment：逐目标输出 polygon/mask，同类多个目标是不同实例
- semseg：逐像素输出类别，同类目标合并成一片区域（语义分割）

本算法用 torchvision 骨干（U-Net / DeepLabV3+ / FCN），训练/推理走
src.train_semseg / src.detect_semseg，不依赖 ultralytics。
"""

from .base import AlgorithmPlugin, Capability, register_algorithm

S_SEMSEG = (
    ("U-Net", "unet", (("ResNet34", "resnet34"), ("ResNet50", "resnet50"))),
    ("DeepLabV3+", "deeplabv3", (("ResNet50", "resnet50"), ("ResNet101", "resnet101"))),
    ("FCN", "fcn", (("ResNet50", "resnet50"),)),
)


def register_semseg_algorithm() -> None:
    register_algorithm(AlgorithmPlugin(
        key="semseg",
        label_key="algo_semseg",
        train_module="src.train_semseg",
        detect_module="src.detect_semseg",
        detect_func="detect_semseg",
        capabilities=(
            Capability(
                task="semseg",
                families=S_SEMSEG,
                dataset_kind="folder",       # images/ + masks/ 数据集根目录
                supports_track=False,
                family_label="semseg_model",
            ),
        ),
    ))
