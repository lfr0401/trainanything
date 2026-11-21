"""CNN 分类算法插件。

与 YOLO classify 同为图像分类任务，但算法完全不同：
- YOLO classify：用 ultralytics 的 YOLO-cls 权重（yolo26n-cls 等）
- CNN classify：用 torchvision 经典卷积网络（ResNet / EfficientNet / MobileNet）

数据集约定与 YOLO classify 一致：分类根目录，每个子文件夹是一个类别
（ImageFolder 结构），GUI 里同样选"数据集根目录"。

新增本算法的意义：验证"任务 × 算法"正交架构——同一个 classify 任务下，
用户在界面上可以自由切换 YOLO / CNN 两种算法，模型清单、执行模块随算法变化。
"""

from .base import AlgorithmPlugin, Capability, register_algorithm

# 家族：(显示名, 模型构造标识, 尺寸档位)
# 模型标识形如 resnet18 / efficientnet_b0 / mobilenet_v3_small，执行模块据此构造网络
S_CNN = (
    ("ResNet", "resnet", (("18", "18"), ("34", "34"), ("50", "50"))),
    ("EfficientNet", "efficientnet", (("B0", "b0"), ("B2", "b2"))),
    ("MobileNet", "mobilenet", (("V3-Small", "v3_small"),)),
)


def register_cnn_algorithm() -> None:
    register_algorithm(AlgorithmPlugin(
        key="cnn",
        label_key="algo_cnn",
        train_module="src.train_cnn",
        detect_module="src.detect_cnn",
        detect_func="detect_cnn",
        capabilities=(
            Capability(
                task="classify",
                families=S_CNN,
                dataset_kind="folder",       # ImageFolder：每个子目录一个类别
                supports_track=False,
                family_label="cnn_model",
            ),
        ),
    ))
