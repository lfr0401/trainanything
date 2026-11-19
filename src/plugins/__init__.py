"""任务插件包。

GUI 从这里读取任务/模型元数据；新增 CV 任务时：
1. 新建模块（参考 yolo.py）调用 register() 注册 TaskPlugin
2. 在本文件导入并调用其 register 函数
无需改动 gui_app.py 的任务/模型定义。
"""

from .base import (
    TaskPlugin,
    register,
    get_plugin,
    all_plugins,
    task_keys,
    model_options_by_task,
    model_name_to_type_by_task,
    supports_track,
    dataset_kind,
)
from .yolo import register_yolo_plugins

# 注册内置插件。后续加任务：import 新模块并调用其 register 函数即可。
register_yolo_plugins()

TASK_KEYS = task_keys()
MODEL_OPTIONS_BY_TASK = model_options_by_task()
MODEL_NAME_TO_TYPE_BY_TASK = model_name_to_type_by_task(MODEL_OPTIONS_BY_TASK)

__all__ = [
    "TaskPlugin", "register", "get_plugin", "all_plugins",
    "TASK_KEYS", "MODEL_OPTIONS_BY_TASK", "MODEL_NAME_TO_TYPE_BY_TASK",
    "supports_track", "dataset_kind",
]
