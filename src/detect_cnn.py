"""CNN 分类推理模块（算法 cnn）。

与 src.detect 的接口保持一致，供 GUI 按算法分发调用：
    detect_cnn(images_folder, model_path, callback, progress_callback,
               conf_threshold, selected_files, tracking, tracker)

模型是 src.train_cnn 产出的 .pt（state_dict + 元信息），不依赖 ultralytics。
结果目录结构（与 YOLO 输出兼容）：
    results/时间戳/images/  原图副本 + 类别标注图
    results/时间戳/labels/  每张图一个 .txt，内容 "cls conf"（与 YOLO-cls 标签一致）
"""

from pathlib import Path
from datetime import datetime
import torch
from PIL import Image, ImageDraw

from .detect import (
    is_valid_image, is_valid_video, normalize_path,
    get_media_files, _find_dataset_root,
)
from .train_cnn import build_model

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _predict_one(model, image_path, img_size, device, class_names):
    from torchvision import transforms

    image = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    x = tf(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        conf, cls_id = torch.max(probs, dim=0)
    return image, int(cls_id.item()), float(conf.item())


def _annotate(image, label, color=(30, 144, 255)):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    text_h = max(20, height // 20)
    draw.rectangle([0, 0, width, text_h], fill=color)
    draw.text((8, max(0, text_h // 4)), label, fill=(255, 255, 255))
    return canvas


def detect_cnn(images_folder, model_path, callback=None, progress_callback=None,
               conf_threshold=0.5, selected_files=None, tracking=False, tracker="bytetrack.yaml"):
    """CNN 分类批量推理；tracking 参数仅为接口兼容，分类不支持跟踪。"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "resnet18")
    num_classes = int(ckpt.get("num_classes", 2))
    img_size = int(ckpt.get("img_size", 640))
    class_names = ckpt.get("class_names", [f"class_{i}" for i in range(num_classes)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_type, num_classes, device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if progress_callback:
        progress_callback(f"CNN model: {model_type} ({num_classes} classes)")
        progress_callback(f"Classes: {', '.join(class_names[:8])}" + (" ..." if len(class_names) > 8 else ""))

    image_files = []
    if selected_files:
        selected = [normalize_path(p) for p in selected_files]
        image_files = [p for p in selected if p.is_file() and is_valid_image(p)]
        base_dir = selected[0].parent if selected else None
    else:
        base_dir = normalize_path(images_folder)
        image_files, _ = get_media_files(base_dir)

    if not image_files:
        message = "No valid images found"
        if progress_callback:
            progress_callback(message)
        raise ValueError(message)

    dataset_root = _find_dataset_root(base_dir)
    if dataset_root is not None:
        output_base = dataset_root.parent / f"{dataset_root.name}_results"
        if progress_callback:
            progress_callback(f"Input is inside dataset {dataset_root}; writing results outside it: {output_base}")
    else:
        output_base = base_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = output_base / "results" / timestamp
    images_out = results_dir / "images"
    labels_out = results_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(f"Output: {results_dir}")

    total = len(image_files)
    for index, path in enumerate(image_files, start=1):
        image, cls_id, conf = _predict_one(model, path, img_size, device, class_names)
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label_line = f"{cls_id} {conf:.6f}"
        annotated = _annotate(image, f"{name} {conf:.2f}")
        annotated.save(images_out / f"{path.stem}.jpg", quality=95)
        (labels_out / f"{path.stem}.txt").write_text(label_line + "\n", encoding="utf-8")
        if progress_callback:
            progress_callback(f"Classifying images {index}/{total} - {name} {conf:.2f}")

    if callback:
        callback(str(results_dir))
    return str(results_dir)
