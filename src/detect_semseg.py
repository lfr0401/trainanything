"""语义分割推理模块（引擎 semseg）。

与 src.detect 的接口保持一致，供 GUI 按引擎分发调用：
    detect_semseg(images_folder, model_path, callback, progress_callback,
                  conf_threshold, selected_files)

模型是 src.train_semseg 产出的 .pt（state_dict + 元信息），不依赖 ultralytics。
结果目录结构：
    results/时间戳/images/  原图 + 彩色 mask 叠图
    results/时间戳/masks/   类别 id 的灰度 mask png（可直接当训练标签）
"""

from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
from PIL import Image

from .detect import (
    is_valid_image, is_valid_video, normalize_path,
    get_media_files, _find_dataset_root,
)

# 固定调色板（0=背景黑色），不同类别给高对比色
_PALETTE = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 128, 255),
    (255, 255, 128), (0, 128, 128), (128, 0, 128),
]


def _color_of(class_id):
    return _PALETTE[class_id % len(_PALETTE)]


def _build_model(model_type, num_classes, device):
    # 复用训练模块的模型构造，保证 train/predict 一致
    from .train_semseg import build_model
    return build_model(model_type, num_classes, device)


def _mask_to_rgb(mask_np, num_classes):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        rgb[mask_np == c] = _color_of(c)
    return Image.fromarray(rgb)


def _overlay(image_rgb, mask_rgb, alpha=0.5):
    return Image.blend(image_rgb.convert("RGB"), mask_rgb.convert("RGB"), alpha)


def _predict_one(model, image_path, img_size, device):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    x = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.array(x).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        if isinstance(out, dict):
            out = out["out"]
        out = torch.nn.functional.interpolate(out, size=(img_size, img_size), mode="bilinear", align_corners=False)
        pred = out.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    # 还原到原图尺寸，用最近邻保持类别 id 不插值
    mask_img = Image.fromarray(pred, mode="L").resize(original_size, Image.Resampling.NEAREST)
    return image, np.array(mask_img)


def detect_semseg(images_folder, model_path, callback=None, progress_callback=None,
                  conf_threshold=0.5, selected_files=None, tracking=False, tracker="bytetrack.yaml"):
    """语义分割批量推理；tracking 参数仅为接口兼容，语义分割不支持跟踪。"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "unet-resnet34")
    num_classes = int(ckpt.get("num_classes", 2))
    img_size = int(ckpt.get("img_size", 640))
    class_names = ckpt.get("class_names", [f"class_{i}" for i in range(num_classes)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type, num_classes, device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if progress_callback:
        progress_callback(f"Semantic segmentation model: {model_type} ({num_classes} classes)")
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
    masks_out = results_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(f"Output: {results_dir}")

    total = len(image_files)
    for index, path in enumerate(image_files, start=1):
        image_rgb, mask_np = _predict_one(model, path, img_size, device)
        stem = path.stem
        # 彩色 mask 叠图供查看
        mask_rgb = _mask_to_rgb(mask_np, num_classes)
        overlay = _overlay(image_rgb, mask_rgb)
        overlay.save(images_out / f"{stem}_mask.jpg", quality=95)
        Image.fromarray(mask_np, mode="L").save(masks_out / f"{stem}.png")
        if progress_callback:
            progress_callback(f"Segmenting images {index}/{total}...")

    if callback:
        callback(str(results_dir))
    return str(results_dir)
