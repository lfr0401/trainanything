import shutil
import os
import cv2
import mimetypes
from pathlib import Path
from typing import List, Union
from datetime import datetime

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

from ultralytics import YOLO

VALID_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ppm',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.PPM'
}

VALID_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv',
    '.MP4', '.AVI', '.MOV', '.MKV', '.WMV', '.FLV'
}

def is_valid_image(file_path: Union[str, Path]) -> bool:
    """Check if a file is a valid image by examining both extension and mime type"""
    try:
        file_path = Path(file_path)
        # Check file extension
        if file_path.suffix.lower() not in {ext.lower() for ext in VALID_IMAGE_EXTENSIONS}:
            return False
            
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith('image/'):
            return False
            
        return True
    except Exception:
        return False

def is_valid_video(file_path: Union[str, Path]) -> bool:
    """Check if a file is a valid video by examining both extension and mime type"""
    try:
        file_path = Path(file_path)
        # Check file extension
        if file_path.suffix.lower() not in {ext.lower() for ext in VALID_VIDEO_EXTENSIONS}:
            return False
            
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith('video/'):
            return False
            
        return True
    except Exception:
        return False

def _find_dataset_root(directory: Path):
    """Return the YOLO dataset root if directory sits inside one, else None.

    A dataset root is detected by the standard images/ + labels/ sibling layout.
    Detection outputs must never be written inside such a tree, otherwise the
    copied images/labels get picked up by the next training run as corrupt data.
    """
    for parent in [directory] + list(directory.parents):
        try:
            if (parent / "images").is_dir() and (parent / "labels").is_dir():
                return parent
        except OSError:
            continue
    return None


def normalize_path(file_path: Union[str, Path]) -> Path:
    """Normalize path to handle Japanese characters and different path formats"""
    try:
        return Path(file_path).resolve()
    except Exception:
        raise ValueError(f"Invalid path: {file_path}")

def _task_supports_track(task):
    """从插件注册表判断 YOLO 算法下该任务是否支持跟踪，避免写死任务列表。"""
    try:
        from src.plugins import supports_track
        return supports_track(task, "yolo")
    except Exception:
        return task in {"detect", "segment", "pose", "obb"}


def _format_label_lines(result, task, conf_threshold, with_track=False):
    """Build YOLO-style label lines for one prediction result, adapted to task.

    detect  : cls conf x1 y1 x2 y2
    segment : cls conf x1 y1 x2 y2 poly_x1 poly_y1 ...   (normalized polygon)
    pose    : cls conf x1 y1 x2 y2 kx1 ky1 kv1 ...       (keypoints + visibility)
    obb     : cls conf x1 y1 x2 y2 cx1 cy1 cx2 cy2 cx3 cy3 cx4 cy4
    classify: cls conf
    开启目标跟踪时，每行末尾追加目标编号 track_id。
    """
    lines = []
    if task == "classify":
        probs = getattr(result, "probs", None)
        if probs is not None and len(probs) > 0:
            conf = float(probs.top1conf)
            if conf >= conf_threshold:
                lines.append(f"{int(probs.top1)} {conf:.4f}\n")
        return lines

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return lines

    masks = getattr(result, "masks", None) if task == "segment" else None
    keypoints = getattr(result, "keypoints", None) if task == "pose" else None
    obb = getattr(result, "obb", None) if task == "obb" else None

    for i, box in enumerate(boxes):
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls = int(box.cls[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        parts = [str(cls), f"{conf:.4f}", str(x1), str(y1), str(x2), str(y2)]
        # 目标跟踪：从 box.id 读取跨帧一致的编号，追加到行尾
        if with_track and getattr(box, "id", None) is not None:
            parts.append(str(int(box.id[0])))

        if task == "segment" and masks is not None and i < len(masks.xyn):
            for px, py in masks.xyn[i]:
                parts.extend([f"{float(px):.6f}", f"{float(py):.6f}"])
        elif task == "pose" and keypoints is not None and i < len(keypoints.xy):
            kpts = keypoints.xy[i]
            kconf = getattr(keypoints, "conf", None)
            for k, (kx, ky) in enumerate(kpts):
                vis = 2
                if kconf is not None and i < len(kconf) and k < len(kconf[i]):
                    vis = 1 if float(kconf[i][k]) >= conf_threshold else 0
                parts.extend([f"{float(kx):.1f}", f"{float(ky):.1f}", str(vis)])
        elif task == "obb" and obb is not None and i < len(obb.xyxyxyxy):
            for px, py in obb.xyxyxyxy[i]:
                parts.extend([f"{float(px):.1f}", f"{float(py):.1f}"])

        lines.append(" ".join(parts) + "\n")
    return lines

def get_media_files(directory: Union[str, Path]) -> tuple[List[Path], List[Path]]:
    """Recursively find all valid images and videos in a directory"""
    directory = normalize_path(directory)
    image_files = []
    video_files = []
    
    try:
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            if is_valid_image(file_path):
                image_files.append(file_path)
            elif is_valid_video(file_path):
                video_files.append(file_path)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return [], []
        
    return sorted(image_files), sorted(video_files)

def process_video(video_path: Path, model, output_dir: Path, conf_threshold: float = 0.5,
                  task: str = "detect", tracking: bool = False, tracker: str = "bytetrack.yaml"):
    """Process a video file and save detection results.

    tracking=True 时用 model.track(persist=True) 做跨帧目标跟踪，标签末尾追加目标 ID。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    video_name = video_path.stem
    video_output_dir = output_dir / f"{timestamp}_{video_name}"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Create video writer
    output_video_path = video_output_dir / f"{video_name}_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress (every 30 frames to avoid excessive printing)
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing video: {progress:.1f}% ({frame_count}/{total_frames} frames)", end="")

        # Detect objects in frame（tracking 时启用跨帧目标跟踪）
        if tracking:
            results = model.track(
                frame, persist=True, tracker=tracker, save=False, conf=conf_threshold, verbose=False
            )
        else:
            results = model.predict(frame, save=False, conf=conf_threshold, verbose=False)

        # Only process frames with detections above threshold
        label_lines = _format_label_lines(results[0], task, conf_threshold, with_track=tracking)
        if label_lines:
            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Save frame with detections
            frame_path = video_output_dir / f"frame_{detection_count:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)

            # Save detection results to txt
            txt_path = video_output_dir / f"frame_{detection_count:04d}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.writelines(label_lines)
            
            detection_count += 1
            
            # Write frame to output video
            out.write(annotated_frame)
        else:
            # Write original frame to video if no detections
            out.write(frame)

        frame_count += 1

    # Clean up
    cap.release()
    out.release()
    
    print(f"\nVideo processing complete. {detection_count} frames with detections saved.")
    print(f"Output video saved to: {output_video_path}")
    
    return video_output_dir

def move_detection_results(source_dir, target_dir):
    source_dir = normalize_path(source_dir)
    target_dir = normalize_path(target_dir)
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in source_dir.iterdir():
        target_file = target_dir / file_path.name
        
        # Remove existing file/directory if it exists
        if target_file.exists():
            if target_file.is_dir():
                shutil.rmtree(str(target_file))
            else:
                target_file.unlink()
        
        # Move the file
        if file_path.is_file():
            shutil.move(str(file_path), str(target_file))
        else:
            shutil.move(str(file_path), str(target_dir))
    
    # Clean up source directory
    shutil.rmtree(str(source_dir))

def _predict_images_one_by_one(model, image_files, images_out, labels_out, conf_threshold, progress_callback, task="detect"):
    """Run inference one image at a time to keep memory usage constant.

    Passing a list of paths to ultralytics makes it load every image into RAM
    at once via LoadPilAndNumpy, which raises MemoryError on large batches.
    Predicting a single path at a time avoids that entirely.

    ``task`` selects the label format: detect / segment / classify / pose / obb.
    """
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    total = len(image_files)
    for index, path in enumerate(image_files, start=1):
        results = model.predict(
            str(path),
            save=False,
            imgsz=640,
            conf=conf_threshold,
            verbose=False,
        )
        for result in results:
            annotated = result.plot()
            stem = path.stem
            cv2.imwrite(str(images_out / f"{stem}.jpg"), annotated)
            lines = _format_label_lines(result, task, conf_threshold)
            with open(labels_out / f"{stem}.txt", "w", encoding="utf-8") as f:
                f.writelines(lines)
        if progress_callback:
            progress_callback(f"Detecting images {index}/{total}...")


def detect_images(images_folder, model_path, callback=None, progress_callback=None, conf_threshold=0.5,
                  selected_files=None, tracking=False, tracker="bytetrack.yaml"):
    model = YOLO(model_path)
    task = getattr(model, "task", "detect") or "detect"
    # 分类任务没有框，无法跟踪；按插件声明屏蔽
    if tracking and not _task_supports_track(task):
        tracking = False
        if progress_callback:
            progress_callback(f"Task '{task}' does not support tracking, disabled.")
    if progress_callback:
        progress_callback(f"Task: {task}" + (" (tracking on)" if tracking else ""))

    image_files = []
    video_files = []
    base_dir = None

    if selected_files:
        # User picked single/multiple files directly.
        selected = [normalize_path(p) for p in selected_files]
        for path in selected:
            if path.is_file() and is_valid_image(path):
                image_files.append(path)
            elif path.is_file() and is_valid_video(path):
                video_files.append(path)
        base_dir = selected[0].parent if selected else None
    else:
        # Scan a whole folder.
        base_dir = normalize_path(images_folder)
        image_files, video_files = get_media_files(base_dir)

    if not image_files and not video_files:
        message = "No valid media files found"
        print(message)
        if progress_callback:
            progress_callback(message)
        raise ValueError(message)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 输入若位于 YOLO 数据集内部（含 images/ + labels/ 的目录树），结果写到数据集
    # 旁边而不是里面，避免检测输出的 6 列标签被下次训练当成 corrupt 数据扫描进去。
    dataset_root = _find_dataset_root(base_dir)
    if dataset_root is not None:
        output_base = dataset_root.parent / f"{dataset_root.name}_results"
        if progress_callback:
            progress_callback(
                f"Input is inside dataset {dataset_root}; writing results outside it: {output_base}"
            )
    else:
        output_base = base_dir
    results_dir = output_base / 'results' / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(f"Output: {results_dir}")

    if image_files:
        if progress_callback:
            progress_callback(f"Processing {len(image_files)} image file(s).")
        _predict_images_one_by_one(
            model, image_files,
            results_dir / 'images', results_dir / 'labels',
            conf_threshold, progress_callback, task,
        )

    for index, video_file in enumerate(video_files, start=1):
        if progress_callback:
            progress_callback(f"Processing video {index}/{len(video_files)}: {video_file.name}")
        process_video(
            video_file, model, results_dir, conf_threshold=conf_threshold,
            task=task, tracking=tracking, tracker=tracker,
        )

    if callback:
        callback(str(results_dir))
