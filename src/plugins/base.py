"""插件注册表：任务 × 算法 正交架构。

设计原则：
- 本包只放**元数据**，不 import torch / ultralytics / cv2，确保 GUI 启动零成本、界面不卡。
- 两个正交维度：
    * 任务 task  —— 做什么：detect / segment / classify / pose / obb / semseg
    * 算法 algorithm —— 用什么做：yolo / semseg / cnn / ...（未来可插拔）
  同一个任务可由多个算法完成，例如 classify 既能用 yolo-cls，也能用 CNN(ResNet)。
- AlgorithmPlugin 声明一个算法引擎（训练/推理执行模块）+ 它在若干任务下的能力 Capability。
- Capability 描述"该算法在该任务下"的模型清单、数据集类型、是否支持跟踪、模型区标题。
- GUI 三级联动：任务 -> 该任务支持的算法 -> 该算法在该任务下的模型。
- 训练/推理分发按 algorithm 查执行模块，不再由 task 反推引擎。

新增算法 = 新建模块 register_algorithm(AlgorithmPlugin(...)) + 在 __init__.py 导入一次
          + 新增对应的 train_xxx.py / detect_xxx.py；主界面代码不用改。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 尺寸档位：((显示名, 后缀), ...)，例如 (("Nano", "n"), ("Small", "s"))
SizeSpec = Tuple[Tuple[str, str], ...]
# 家族：(家族显示名, 模型名前缀, 尺寸档位)
FamilySpec = Tuple[str, str, SizeSpec]


@dataclass(frozen=True)
class Capability:
    """某算法在某个任务下的一组能力声明。

    task:           任务 key，对应 gui_text 的 task_<key>
    families:       可选模型家族
    model_suffix:   模型名后缀（YOLO 权重用 -seg / -cls 等；其它算法一般为空）
    dataset_kind:   'file' 选 data.yaml；'folder' 选数据集根目录
    supports_track: 该算法+任务组合是否支持视频/摄像头目标跟踪
    family_label:   模型区标题文案键（yolo_model / semseg_model / cnn_model ...）
    """

    task: str
    families: Tuple[FamilySpec, ...]
    model_suffix: str = ""
    dataset_kind: str = "file"
    supports_track: bool = False
    family_label: str = "yolo_model"


@dataclass(frozen=True)
class AlgorithmPlugin:
    """一个算法引擎插件。

    key:           算法 key，如 yolo / semseg / cnn，对应 gui_text 的 algo_<key>
    label_key:     算法下拉显示文案键
    train_module:  训练执行模块（python -m 形式）
    detect_module: 图片/视频推理模块（python 导入路径）
    detect_func:   推理模块内的入口函数名（如 detect_images / detect_cnn）
    capabilities:  该算法支持的任务及每个任务下的模型配置
    """

    key: str
    label_key: str
    train_module: str
    detect_module: str
    detect_func: str
    capabilities: Tuple[Capability, ...]


_ALGORITHMS: Dict[str, AlgorithmPlugin] = {}


def register_algorithm(plugin: AlgorithmPlugin) -> None:
    """注册一个算法插件；重复 key 直接覆盖，便于扩展。"""
    _ALGORITHMS[plugin.key] = plugin


def all_algorithms() -> List[AlgorithmPlugin]:
    """按注册顺序返回全部算法。"""
    return list(_ALGORITHMS.values())


def algorithm_keys() -> List[str]:
    return [a.key for a in all_algorithms()]


def get_algorithm(key: str) -> Optional[AlgorithmPlugin]:
    return _ALGORITHMS.get(key)


def task_keys() -> List[str]:
    """全部任务 key，按首次出现顺序去重（任务由各算法的能力表聚合而来）。"""
    seen: List[str] = []
    for algo in all_algorithms():
        for cap in algo.capabilities:
            if cap.task not in seen:
                seen.append(cap.task)
    return seen


def algorithms_for_task(task: str) -> List[str]:
    """支持该任务的算法 key 列表（按注册顺序）。"""
    return [a.key for a in all_algorithms() if any(c.task == task for c in a.capabilities)]


def _capability(task: str, algorithm: str) -> Optional[Capability]:
    algo = get_algorithm(algorithm)
    if algo is None:
        return None
    for cap in algo.capabilities:
        if cap.task == task:
            return cap
    return None


def model_options(task: str, algorithm: str) -> List[Tuple[str, str]]:
    """[(模型显示名, 模型标识), ...]；无匹配时返回空列表。"""
    cap = _capability(task, algorithm)
    if cap is None:
        return []
    options: List[Tuple[str, str]] = []
    for family_label, family_prefix, sizes in cap.families:
        for size_label, size_suffix in sizes:
            options.append((
                f"{family_label}-{size_label}{cap.model_suffix}",
                f"{family_prefix}{size_suffix}{cap.model_suffix}",
            ))
    return options


def model_type_of(task: str, algorithm: str, display_name: str) -> str:
    """模型显示名 -> 模型标识；找不到返回空串。"""
    for display, value in model_options(task, algorithm):
        if display == display_name:
            return value
    return ""


def dataset_kind(task: str, algorithm: str) -> str:
    cap = _capability(task, algorithm)
    return cap.dataset_kind if cap else "file"


def supports_track(task: str, algorithm: str) -> bool:
    cap = _capability(task, algorithm)
    return bool(cap and cap.supports_track)


def family_label_of(task: str, algorithm: str) -> str:
    cap = _capability(task, algorithm)
    return cap.family_label if cap else "yolo_model"


def algorithm_label_key(algorithm: str) -> str:
    algo = get_algorithm(algorithm)
    return algo.label_key if algo else f"algo_{algorithm}"


def train_module_of(algorithm: str) -> str:
    algo = get_algorithm(algorithm)
    return algo.train_module if algo else "src.train"


def detect_module_of(algorithm: str) -> str:
    algo = get_algorithm(algorithm)
    return algo.detect_module if algo else "src.detect"


def detect_func_of(algorithm: str) -> str:
    algo = get_algorithm(algorithm)
    return algo.detect_func if algo else "detect_images"


def registered_algorithms() -> Tuple[str, ...]:
    """返回所有已注册算法 key，供权重探测时校验。"""
    return tuple(_ALGORITHMS.keys())
