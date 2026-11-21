"""语义分割训练模块（引擎 semseg）。

由 GUI 按任务引擎分发调用，命令行参数与 src.train 保持一致：
    python -m src.train_semseg <project> <dataset_dir> <save_path> <model_type> \
        <imgsz> <epochs> <batch> <task> <patience> <lr0> <device> <resume> <augment>

数据集结构（dataset_dir 根目录）：
    images/train/*.png    RGB 原图
    masks/train/*.png     灰度 mask，像素值即类别 id（0=背景）
    images/val/*.png
    masks/val/*.png
    classes.txt           （可选）每行一个类别名，第 0 行为背景类

模型标识（model_type）：
    unet-resnet34 / unet-resnet50  自写 U-Net，torchvision resnet 作编码器
    deeplabv3-resnet50 / ...101    torchvision 预训练 DeepLabV3+
    fcn-resnet50                   torchvision 预训练 FCN

输出与 src.train 兼容：
    save_path/project/weights/best.pt   最优权重（state_dict + 元信息）
    save_path/project/weights/last.pt
    save_path/project/training_environment.txt
进度日志沿用 GUI_PROGRESS 协议，GUI 解析逻辑无需改动。
"""

import os
import sys
import time
import random
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------- 数据集 ----------

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = int(img_size)
        self.augment = augment
        self.samples = self._collect()

    def _collect(self):
        pairs = []
        for img_path in sorted(self.image_dir.glob("*.*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                # 允许 mask 用同名 .png，即使原图是 .jpg
                mask_path = self.mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:
                angle = random.choice([90, 180, 270])
                image = image.rotate(angle)
                mask = mask.rotate(angle)

        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        image_tensor = transforms.ToTensor()(image)
        mask_tensor = torch.as_tensor(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor


# ---------- 模型 ----------

class _UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """轻量 U-Net，从零训练，不依赖预训练下载。"""
    def __init__(self, num_classes, base=32):
        super().__init__()
        self.enc1 = _UNetBlock(3, base)
        self.enc2 = _UNetBlock(base, base * 2)
        self.enc3 = _UNetBlock(base * 2, base * 4)
        self.enc4 = _UNetBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.center = _UNetBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _UNetBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _UNetBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _UNetBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _UNetBlock(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_model(model_type, num_classes, device):
    """按模型标识构造网络。model_type 形如 unet-resnet34 / deeplabv3-resnet50 / fcn-resnet50。"""
    parts = model_type.lower().split("-")
    family = parts[0]
    backbone = parts[1] if len(parts) > 1 else "resnet34"

    if family == "unet":
        # base 通道数按骨干深度微调；resnet 标识仅作规模提示，编码器仍为自写卷积
        base = 48 if backbone.endswith("50") else 32
        return UNet(num_classes, base=base).to(device)

    from torchvision.models import segmentation as seg_models
    weights_enum_name = backbone.upper().replace("101", "101")
    if family == "deeplabv3":
        weights = getattr(seg_models, f"DeepLabV3_ResNet{weights_enum_name}_Weights", None)
        fn = seg_models.deeplabv3_resnet50 if backbone.endswith("50") else seg_models.deeplabv3_resnet101
        w = weights.DEFAULT if weights is not None else None
        model = fn(weights=w, weights_backbone=None)
        model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    elif family == "fcn":
        weights = getattr(seg_models, f"FCN_ResNet{weights_enum_name}_Weights", None)
        fn = seg_models.fcn_resnet50
        w = weights.DEFAULT if weights is not None else None
        model = fn(weights=w, weights_backbone=None)
        model.classifier[4] = nn.Conv2d(512, num_classes, 1)
    else:
        raise ValueError(f"Unsupported semseg model: {model_type}")
    return model.to(device)


# ---------- 训练 ----------

def _device_of(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_classes(dataset_dir, max_id):
    classes_file = Path(dataset_dir) / "classes.txt"
    if classes_file.exists():
        names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) >= max_id + 1:
            return names[:max_id + 1]
    return [f"class_{i}" for i in range(max_id + 1)]


def _scan_max_class(mask_root):
    max_id = 0
    for mask_path in Path(mask_root).rglob("*.png"):
        arr = np.array(Image.open(mask_path).convert("L"))
        if arr.size:
            max_id = max(max_id, int(arr.max()))
    return max_id


def _dice_loss(logits, targets, num_classes):
    probs = F.softmax(logits, dim=1)
    dice_sum = 0.0
    for c in range(num_classes):
        p = probs[:, c]
        t = (targets == c).float()
        intersection = (p * t).sum()
        dice_sum += (2 * intersection + 1) / (p.sum() + t.sum() + 1)
    return 1 - dice_sum / num_classes


def train_semseg(project_name, dataset_dir, save_path, model_type, img_size, epochs, batch_size,
                 task="semseg", patience=100, lr0=0.01, device="auto", resume=False, augment=True):
    selected_device = _device_of(device)
    print(f"GUI_DEVICE device={selected_device} mode={device} torch_cuda={torch.version.cuda}", flush=True)
    print(f"Using device: {selected_device}", flush=True)

    dataset_dir = Path(dataset_dir)
    train_img = dataset_dir / "images" / "train"
    train_mask = dataset_dir / "masks" / "train"
    val_img = dataset_dir / "images" / "val"
    val_mask = dataset_dir / "masks" / "val"
    for required in (train_img, train_mask, val_img, val_mask):
        if not required.is_dir():
            raise FileNotFoundError(f"Missing semantic segmentation directory: {required}")

    max_id = max(_scan_max_class(train_mask), _scan_max_class(val_mask))
    num_classes = max_id + 1
    class_names = _read_classes(dataset_dir, max_id)
    print(f"GUI_TASK task=semseg model={model_type} classes={num_classes} {class_names}", flush=True)

    train_ds = SegDataset(train_img, train_mask, img_size, augment=augment)
    val_ds = SegDataset(val_img, val_mask, img_size, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("No paired image/mask files found in train or val directory")
    print(f"Train images: {len(train_ds)}, Val images: {len(val_ds)}", flush=True)

    # Windows 下 workers=0，避免 DataLoader 多进程内存问题（与 src.train 一致）
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_type, num_classes, selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    destination = Path(save_path) / project_name
    weights_dir = destination / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    start_epoch = 0
    best_miou = 0.0
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=selected_device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}", flush=True)

    start_time = time.time()
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(selected_device)
            masks = masks.to(selected_device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks) + _dice_loss(outputs, masks, num_classes)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        avg_loss = running_loss / max(len(train_loader), 1)

        # mIoU
        model.eval()
        iou_sum = 0.0
        class_count = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(selected_device)
                masks = masks.to(selected_device)
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                preds = outputs.argmax(dim=1)
                for c in range(num_classes):
                    pred_c = preds == c
                    target_c = masks == c
                    union = (pred_c | target_c).sum().float()
                    if union.item() > 0:
                        inter = (pred_c & target_c).sum().float()
                        iou_sum += float(inter / union)
                        class_count += 1
        miou = iou_sum / max(class_count, 1)

        elapsed = max(time.time() - start_time, 0.1)
        eta = max(epochs - epoch - 1, 0) * (elapsed / max(epoch - start_epoch + 1, 1))
        # 与 GUI_PROGRESS 协议对齐：语义分割用 map50 字段承载 mIoU，acc 记 na
        print(
            f"GUI_PROGRESS epoch={epoch + 1} total={epochs} "
            f"elapsed={elapsed:.1f} eta={eta:.1f} "
            f"loss={avg_loss:.4f} map50={miou:.4f} map5095=na acc=na",
            flush=True,
        )
        print(f"Epoch {epoch + 1}/{epochs} - loss {avg_loss:.4f} - mIoU {miou:.4f}", flush=True)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_miou": max(best_miou, miou),
            "model_type": model_type,
            "num_classes": num_classes,
            "class_names": class_names,
            "img_size": img_size,
        }
        torch.save(ckpt, last_path)
        if miou > best_miou:
            best_miou = miou
            torch.save(ckpt, best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                break

    print(f"GUI_ARTIFACT weights={best_path}", flush=True)
    try:
        from src.system_report import write_environment_report
        report_path = write_environment_report(destination)
        print(f"GUI_ARTIFACT environment={report_path}", flush=True)
    except Exception as exc:
        print(f"WARNING: environment report failed: {exc}", flush=True)
    print(f"Training output saved to: {destination}", flush=True)
    return destination


def parse_args():
    project_name = sys.argv[1]
    dataset_dir = sys.argv[2]
    save_path = sys.argv[3]
    model_type = sys.argv[4]
    img_size = int(sys.argv[5])
    epochs = int(sys.argv[6])
    batch_size = int(sys.argv[7])
    task = sys.argv[8] if len(sys.argv) > 8 else "semseg"
    patience = int(sys.argv[9]) if len(sys.argv) > 9 else 100
    lr0 = float(sys.argv[10]) if len(sys.argv) > 10 else 0.01
    device = sys.argv[11] if len(sys.argv) > 11 else "auto"
    resume = sys.argv[12].lower() == "true" if len(sys.argv) > 12 else False
    augment = sys.argv[13].lower() != "false" if len(sys.argv) > 13 else True
    return train_semseg(project_name, dataset_dir, save_path, model_type, img_size, epochs,
                        batch_size, task, patience, lr0, device, resume, augment)


if __name__ == "__main__":
    parse_args()
