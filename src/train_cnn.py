"""CNN 分类训练模块（算法 cnn）。

由 GUI 按算法分发调用，命令行参数与 src.train 完全一致：
    python -m src.train_cnn <project> <dataset_dir> <save_path> <model_type> \
        <imgsz> <epochs> <batch> <task> <patience> <lr0> <device> <resume> <augment>

数据集结构（dataset_dir 根目录，ImageFolder 约定）：
    train/<类别名>/*.jpg    每个子目录一个类别
    val/<类别名>/*.jpg
    classes.txt            （可选）与 train.py 兼容，实际类别取自子目录名

模型标识（model_type，与 src/plugins/cnn.py 的家族声明一致）：
    resnet18 / resnet34 / resnet50
    efficientnet_b0 / efficientnet_b2
    mobilenet_v3_small

输出与 src.train 兼容：
    save_path/project/weights/best.pt   最优权重（state_dict + 元信息）
    save_path/project/weights/last.pt
    save_path/project/training_environment.txt
进度日志沿用 GUI_PROGRESS 协议；分类任务 acc 字段写 Top1，map50/map5095 写 na。
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 模型 ----------

def build_model(model_type, num_classes, device):
    """按模型标识构造网络，首用会自动下载 torchvision 预训练权重（失败则改用随机初始化）。"""
    from torchvision import models

    key = model_type.lower()
    try:
        if key.startswith("resnet"):
            weights = models.ResNet18_Weights.DEFAULT if "18" in key else (
                models.ResNet34_Weights.DEFAULT if "34" in key else models.ResNet50_Weights.DEFAULT
            )
            model = getattr(models, f"resnet{'18' if '18' in key else '34' if '34' in key else '50'}")(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif key.startswith("efficientnet"):
            weights = models.EfficientNet_B0_Weights.DEFAULT if "b0" in key else models.EfficientNet_B2_Weights.DEFAULT
            model = getattr(models, f"efficientnet_{'b0' if 'b0' in key else 'b2'}")(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif key.startswith("mobilenet"):
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported CNN model: {model_type}")
    except Exception as exc:
        # 离线环境下载权重失败时仍可训练（从零开始），保证流程可用
        print(f"WARNING: pretrained weights unavailable ({exc}); training from scratch.", flush=True)
        return build_from_scratch(key, num_classes, device)
    return model.to(device)


def build_from_scratch(key, num_classes, device):
    from torchvision import models
    if key.startswith("resnet"):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key.startswith("efficientnet"):
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif key.startswith("mobilenet"):
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported CNN model: {model_type}")
    return model.to(device)


# ---------- 训练 ----------

def _device_of(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_transforms(img_size, augment):
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss_sum += float(criterion(outputs, labels).item()) * labels.size(0)
        correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return (loss_sum / max(total, 1)), (correct / max(total, 1))


def train_cnn(project_name, dataset_dir, save_path, model_type, img_size, epochs, batch_size,
              task="classify", patience=100, lr0=0.01, device="auto", resume=False, augment=True):
    _seed_everything(0)
    selected_device = _device_of(device)
    print(f"GUI_DEVICE device={selected_device} mode={device} torch_cuda={torch.version.cuda}", flush=True)
    print(f"Using device: {selected_device}", flush=True)

    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    for required in (train_dir, val_dir):
        if not required.is_dir():
            raise FileNotFoundError(f"Missing classification directory: {required}")

    train_tf, val_tf = _make_transforms(int(img_size), augment)
    try:
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    except FileNotFoundError as exc:
        raise ValueError(
            "CNN 分类要求 ImageFolder 结构：train/<类别>/*.jpg 与 val/<类别>/*.jpg"
        ) from exc

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"GUI_TASK task=classify algorithm=cnn model={model_type} classes={num_classes} {class_names}", flush=True)
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
    best_acc = 0.0
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=selected_device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}", flush=True)

    start_time = time.time()
    patience_counter = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(selected_device)
            labels = labels.to(selected_device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
        avg_loss = running_loss / max(len(train_ds), 1)
        _, val_acc = _evaluate(model, val_loader, selected_device)

        elapsed = max(time.time() - start_time, 0.1)
        eta = max(epochs - epoch - 1, 0) * (elapsed / max(epoch - start_epoch + 1, 1))
        # GUI_PROGRESS 协议：分类任务把 Top1 写入 acc，map50/map5095 记 na
        print(
            f"GUI_PROGRESS epoch={epoch + 1} total={epochs} "
            f"elapsed={elapsed:.1f} eta={eta:.1f} "
            f"loss={avg_loss:.4f} map50=na map5095=na acc={val_acc:.4f}",
            flush=True,
        )
        print(f"Epoch {epoch + 1}/{epochs} - loss {avg_loss:.4f} - Top1 {val_acc:.4f}", flush=True)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_acc": max(best_acc, val_acc),
            "algorithm": "cnn",
            "model_type": model_type,
            "num_classes": num_classes,
            "class_names": class_names,
            "img_size": int(img_size),
        }
        torch.save(ckpt, last_path)
        if val_acc > best_acc:
            best_acc = val_acc
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
    task = sys.argv[8] if len(sys.argv) > 8 else "classify"
    patience = int(sys.argv[9]) if len(sys.argv) > 9 else 100
    lr0 = float(sys.argv[10]) if len(sys.argv) > 10 else 0.01
    device = sys.argv[11] if len(sys.argv) > 11 else "auto"
    resume = sys.argv[12].lower() == "true" if len(sys.argv) > 12 else False
    augment = sys.argv[13].lower() != "false" if len(sys.argv) > 13 else True
    return train_cnn(project_name, dataset_dir, save_path, model_type, img_size, epochs,
                     batch_size, task, patience, lr0, device, resume, augment)


if __name__ == "__main__":
    parse_args()
