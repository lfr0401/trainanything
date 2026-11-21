"""CV 算法插件包（任务 × 算法 正交架构）。

GUI 从这里读取任务/算法/模型元数据；新增算法时：
1. 新建模块（参考 cnn.py）调用 register_algorithm() 声明 AlgorithmPlugin
2. 新建对应的 src/train_xxx.py 与 src/detect_xxx.py
3. 在本文件导入并调用其 register 函数
无需改动 gui_app.py 的任务/算法/模型定义。
"""

from .base import (
    AlgorithmPlugin, Capability, register_algorithm,
    all_algorithms, algorithm_keys, get_algorithm, task_keys, algorithms_for_task,
    model_options, model_type_of, dataset_kind, supports_track,
    family_label_of, algorithm_label_key, train_module_of, detect_module_of,
    detect_func_of, registered_algorithms,
)
from .yolo import register_yolo_algorithm
from .semseg import register_semseg_algorithm
from .cnn import register_cnn_algorithm

# 注册内置算法。后续加算法：import 新模块并调用其 register 函数即可。
register_yolo_algorithm()
register_semseg_algorithm()
register_cnn_algorithm()

TASK_KEYS = task_keys()                       # 全部任务（GUI 任务下拉）
ALGORITHM_KEYS = algorithm_keys()             # 全部算法（文档/调试）

__all__ = [
    "AlgorithmPlugin", "Capability", "register_algorithm",
    "all_algorithms", "algorithm_keys", "get_algorithm",
    "TASK_KEYS", "ALGORITHM_KEYS", "task_keys", "algorithms_for_task",
    "model_options", "model_type_of", "dataset_kind", "supports_track",
    "family_label_of", "algorithm_label_key", "train_module_of", "detect_module_of",
    "detect_func_of", "registered_algorithms",
]
