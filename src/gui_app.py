import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

from src.gui_text import TEXT


# 各家族的尺寸档位（显示后缀, 权重后缀）
_MODEL_SIZES = {
    "yolo26": [("Nano", "n"), ("Small", "s"), ("Medium", "m"), ("Large", "l"), ("ExtraLarge", "x")],
    "yolo12": [("Nano", "n"), ("Small", "s"), ("Medium", "m"), ("Large", "l"), ("ExtraLarge", "x")],
    "yolo11": [("Nano", "n"), ("Small", "s"), ("Medium", "m"), ("Large", "l"), ("ExtraLarge", "x")],
    "yolov10": [("Nano", "n"), ("Small", "s"), ("Medium", "m"), ("Balanced", "b"), ("Large", "l"), ("ExtraLarge", "x")],
    "yolov9": [("Compact", "c"), ("Enhanced", "e")],
    "yolov8": [("Nano", "n"), ("Small", "s"), ("Medium", "m"), ("Large", "l"), ("ExtraLarge", "x")],
}
_FAMILY_LABELS = {
    "yolo26": "YOLO26", "yolo12": "YOLO12", "yolo11": "YOLO11",
    "yolov10": "YOLOv10", "yolov9": "YOLOv9", "yolov8": "YOLOv8",
}
# 各任务支持的家族 + 权重名后缀（依据 ultralytics 官方内置模型配置）
TASK_KEYS = ["detect", "segment", "classify", "pose", "obb"]
_TASK_FAMILIES = {
    "detect": (["yolo26", "yolo12", "yolo11", "yolov10", "yolov9", "yolov8"], ""),
    "segment": (["yolo26", "yolo11", "yolov9", "yolov8"], "-seg"),
    "classify": (["yolo26", "yolo11", "yolov8"], "-cls"),
    "pose": (["yolo26", "yolo11", "yolov8"], "-pose"),
    "obb": (["yolo26", "yolo11", "yolov8"], "-obb"),
}


def build_model_options(task):
    families, suffix = _TASK_FAMILIES.get(task, _TASK_FAMILIES["detect"])
    options = []
    for family in families:
        for size_label, size_suffix in _MODEL_SIZES[family]:
            options.append((
                f"{_FAMILY_LABELS[family]}-{size_label}{suffix}",
                f"{family}{size_suffix}{suffix}",
            ))
    return options


MODEL_OPTIONS_BY_TASK = {task: build_model_options(task) for task in TASK_KEYS}
MODEL_NAME_TO_TYPE_BY_TASK = {task: dict(opts) for task, opts in MODEL_OPTIONS_BY_TASK.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

COLORS = {
    "bg": ("#f4f7fb", "#0d1117"),
    "sidebar": ("#eef2f7", "#111827"),
    "panel": ("#ffffff", "#161b22"),
    "panel_alt": ("#f8fafc", "#1f2937"),
    "line": ("#cbd5e1", "#2f3b4a"),
    "text": ("#0f172a", "#e5edf5"),
    "muted": ("#475569", "#9aa7b7"),
    "log_bg": ("#ffffff", "#05070a"),
    "log_text": ("#0f172a", "#d7e4f2"),
    "preview_bg": ("#e5e7eb", "#05070a"),
    "accent": "#2dd4bf",
    "accent_hover": "#14b8a6",
    "blue": "#38bdf8",
    "blue_hover": "#0ea5e9",
    "warning": "#f59e0b",
    "warning_hover": "#d97706",
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
    "success": "#22c55e",
}

PROGRESS_RE = re.compile(
    r"GUI_PROGRESS\s+epoch=(?P<epoch>\d+)\s+total=(?P<total>\d+)\s+"
    r"elapsed=(?P<elapsed>[\d.]+)\s+eta=(?P<eta>[\d.]+)"
    r"(?:\s+loss=(?P<loss>[\d.]+|na))?"
    r"(?:\s+map50=(?P<map50>[\d.]+|na))?"
    r"(?:\s+map5095=(?P<map5095>[\d.]+|na))?"
    r"(?:\s+acc=(?P<acc>[\d.]+|na))?"
)

DEVICE_OPTIONS = ("auto", "cuda", "cpu")


def normalize_path(path):
    if not path:
        return ""
    return str(Path(path).resolve())


def format_duration(seconds, language):
    seconds = max(float(seconds), 0.0)
    if language == "zh":
        if seconds < 60:
            return "不足1分钟"
        minutes = int((seconds + 59) // 60)
        if minutes < 60:
            return f"约{minutes}分钟"
        hours = minutes // 60
        rem = minutes % 60
        return f"约{hours}小时{rem}分"
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return "1分未満" if language == "ja" else "<1 min"
    minutes = int((seconds + 59) // 60)
    if minutes < 60:
        return f"約{minutes}分" if language == "ja" else f"about {minutes} min"
    hours = minutes // 60
    rem = minutes % 60
    if language == "ja":
        return f"約{hours}時間{rem}分"
    return f"about {hours}h {rem}m"


def resolve_color(color, appearance_mode):
    if isinstance(color, tuple):
        return color[0] if appearance_mode == "Light" else color[1]
    return color


def shorten_path(path, limit=56):
    if not path:
        return ""
    text = str(path)
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3):]


class YoloGuiApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        ctk.set_widget_scaling(1.0)

        self.root = ctk.CTk()
        self.root.title("YOLO 训练与检测")
        self.root.minsize(1180, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.language_var = ctk.StringVar(value="zh")
        self.appearance_var = ctk.StringVar(value="Dark")
        self.ui_scale_var = ctk.StringVar(value="100")
        self.selected_task_var = ctk.StringVar(value="detect")
        self.selected_model_var = ctk.StringVar(value=MODEL_OPTIONS_BY_TASK["detect"][0][0])
        self.camera_device_var = ctk.StringVar(value="")

        self.current_view = "train"
        self.busy = False
        self.train_process = None
        self.training_stop_requested = False
        self.training_started_at = None
        self.camera_detection = None
        self.image_paths = []
        self.current_image_index = 0
        self.preview_image = None

        self.data_yaml_path = ""
        self.model_save_path = ""
        self.detection_images_folder_path = ""
        self.detection_selected_files = []
        self.detection_model_path = ""
        self.detection_save_dir = ""
        self.detection_confidence = "0.5"
        self.camera_confidence = "0.5"
        self.camera_devices = []
        self.camera_device_labels = {}

        self.train_state = {
            "project_name": "",
            "input_size": "640",
            "epochs": "100",
            "batch_size": "16",
            "patience": "100",
            "lr0": "0.01",
            "device": "auto",
            "resume": False,
            "augment": True,
        }
        self.resume_var = tk.BooleanVar(value=False)
        self.augment_var = tk.BooleanVar(value=True)

        self.queue = Queue()
        self.nav_buttons = {}
        self.sidebar_busy_widgets = []
        self.main_lock_widgets = []
        self.path_value_labels = {}
        self.left_panel_width = None

        self._build_shell()
        self.show_view("train")
        self.root.after(100, self.process_queue)
        self.root.after(0, self.maximize_window)

    def t(self, key):
        language = self.language_var.get()
        return TEXT.get(language, TEXT["en"]).get(key, TEXT["en"].get(key, key))

    def c(self, key):
        return resolve_color(COLORS[key], self.appearance_var.get())

    def maximize_window(self):
        try:
            self.root.state("zoomed")
        except tk.TclError:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            width = int(screen_w * 0.92)
            height = int(screen_h * 0.9)
            x = max((screen_w - width) // 2, 0)
            y = max((screen_h - height) // 2, 0)
            self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _build_shell(self):
        self.paned = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bd=0,
            sashwidth=7,
            sashrelief="flat",
            opaqueresize=True,
            bg=self.c("line"),
        )
        self.paned.pack(fill="both", expand=True)

        self.sidebar = ctk.CTkFrame(self.paned, width=310, corner_radius=0, fg_color=COLORS["sidebar"])
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.paned.add(self.sidebar, minsize=240, width=310)

        self.main = ctk.CTkFrame(self.paned, corner_radius=0, fg_color=COLORS["bg"])
        self.paned.add(self.main, minsize=720)
        self._build_sidebar()

    def _build_sidebar(self):
        for child in self.sidebar.winfo_children():
            child.destroy()

        self.sidebar_busy_widgets = []
        self.sidebar.grid_rowconfigure(3, weight=1)

        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=16, pady=(18, 14))
        brand.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            brand,
            text=self.t("app_title"),
            font=("Segoe UI", 20, "bold"),
            text_color=COLORS["text"],
            anchor="w",
            wraplength=220,
        ).grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            brand,
            text=self.t("app_badge"),
            font=("Segoe UI", 12),
            text_color=COLORS["muted"],
            anchor="w",
            wraplength=220,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

        nav = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", padx=14, pady=(6, 16))
        nav.grid_columnconfigure(0, weight=1)
        nav_items = [
            ("train", "nav_train"),
            ("detect", "nav_detect"),
            ("camera", "nav_camera"),
        ]
        for row, (view, label_key) in enumerate(nav_items):
            selected = self.current_view == view
            button = ctk.CTkButton(
                nav,
                text=self.t(label_key),
                height=44,
                corner_radius=8,
                anchor="w",
                fg_color=COLORS["accent"] if selected else "transparent",
                hover_color=COLORS["accent_hover"],
                border_color=COLORS["line"],
                border_width=1,
                text_color="#071013" if selected else COLORS["text"],
                font=("Segoe UI", 15, "bold"),
                command=lambda target=view: self.show_view(target),
            )
            button.grid(row=row, column=0, sticky="ew", pady=5)
            self.nav_buttons[view] = button

        status = ctk.CTkFrame(self.sidebar, fg_color=COLORS["panel"], corner_radius=8)
        status.grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 12))
        status.grid_columnconfigure(0, weight=1)
        self.sidebar_status = ctk.CTkLabel(
            status,
            text=self.t("busy") if self.busy else self.t("ready"),
            text_color=COLORS["warning"] if self.busy else COLORS["accent"],
            font=("Segoe UI", 13, "bold"),
            anchor="w",
        )
        self.sidebar_status.grid(row=0, column=0, sticky="ew", padx=12, pady=10)

        settings = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        settings.grid(row=4, column=0, sticky="ew", padx=14, pady=(12, 16))
        settings.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(settings, text=self.t("appearance"), anchor="w", text_color=COLORS["muted"]).grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )
        mode_row = ctk.CTkFrame(settings, fg_color="transparent")
        mode_row.grid(row=1, column=0, sticky="ew")
        dark_radio = ctk.CTkRadioButton(
            mode_row,
            text=self.t("dark"),
            variable=self.appearance_var,
            value="Dark",
            text_color=COLORS["text"],
            command=lambda: self.change_appearance("Dark"),
        )
        dark_radio.pack(side="left", padx=(0, 12))
        self.sidebar_busy_widgets.append(dark_radio)
        light_radio = ctk.CTkRadioButton(
            mode_row,
            text=self.t("light"),
            variable=self.appearance_var,
            value="Light",
            text_color=COLORS["text"],
            command=lambda: self.change_appearance("Light"),
        )
        light_radio.pack(side="left")
        self.sidebar_busy_widgets.append(light_radio)

        ctk.CTkLabel(settings, text=self.t("scale"), anchor="w", text_color=COLORS["muted"]).grid(
            row=2, column=0, sticky="ew", pady=(14, 6)
        )
        scale_row = ctk.CTkFrame(settings, fg_color="transparent")
        scale_row.grid(row=3, column=0, sticky="ew")
        scale_row.grid_columnconfigure(0, weight=1)
        self.scale_entry = ctk.CTkEntry(
            scale_row,
            textvariable=self.ui_scale_var,
            placeholder_text=self.t("scale_placeholder"),
            height=34,
            fg_color=COLORS["panel_alt"],
            border_color=COLORS["line"],
            text_color=COLORS["text"],
        )
        self.scale_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.scale_entry.bind("<Return>", lambda _event: self.apply_ui_scale())
        self.sidebar_busy_widgets.append(self.scale_entry)
        scale_button = ctk.CTkButton(
            scale_row,
            text=self.t("apply"),
            width=70,
            height=34,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            command=self.apply_ui_scale,
        )
        scale_button.grid(row=0, column=1, sticky="e")
        self.sidebar_busy_widgets.append(scale_button)

        ctk.CTkLabel(
            self.sidebar,
            text="SpreadKnowledge 2026",
            text_color=COLORS["muted"],
            font=("Segoe UI", 11),
        ).grid(row=5, column=0, sticky="sw", padx=18, pady=(0, 12))

        self.set_busy(self.busy)

    def change_language(self):
        if self.busy:
            messagebox.showinfo(self.t("info"), self.t("blocked"))
            return
        self.capture_view_state()
        self._build_sidebar()
        self.show_view(self.current_view, force=True)

    def change_appearance(self, mode):
        if self.busy:
            messagebox.showinfo(self.t("info"), self.t("blocked"))
            return
        self.appearance_var.set(mode)
        ctk.set_appearance_mode(mode)
        if hasattr(self, "paned"):
            self.paned.configure(bg=self.c("line"))

    def apply_ui_scale(self):
        if self.busy:
            messagebox.showinfo(self.t("info"), self.t("blocked"))
            return

        raw_value = self.ui_scale_var.get().strip().replace("%", "")
        try:
            scale_percent = int(raw_value)
        except ValueError:
            messagebox.showwarning(self.t("invalid_title"), self.t("invalid_number"))
            return

        scale_percent = min(max(scale_percent, 70), 170)
        self.ui_scale_var.set(str(scale_percent))
        ctk.set_widget_scaling(scale_percent / 100)
        self.capture_view_state()
        self._build_sidebar()
        self.show_view(self.current_view, force=True)

    def set_widgets_enabled(self, widgets, enabled):
        state = "normal" if enabled else "disabled"
        for widget in widgets:
            try:
                if widget is not None and widget.winfo_exists():
                    widget.configure(state=state)
            except (AttributeError, tk.TclError, ValueError):
                pass

    def set_busy(self, busy):
        self.busy = busy
        for button in self.nav_buttons.values():
            button.configure(state="disabled" if busy else "normal")
        self.set_widgets_enabled(self.sidebar_busy_widgets, not busy)
        self.set_widgets_enabled(self.main_lock_widgets, not busy)
        if hasattr(self, "sidebar_status") and self.sidebar_status.winfo_exists():
            self.sidebar_status.configure(
                text=self.t("busy") if busy else self.t("ready"),
                text_color=COLORS["warning"] if busy else COLORS["accent"],
            )

    def show_view(self, view, force=False):
        if self.busy and not force and view != self.current_view:
            messagebox.showinfo(self.t("info"), self.t("blocked"))
            return

        self.capture_view_state()
        self.current_view = view
        self.clear_main()
        self._clear_key_bindings()
        self._build_sidebar()

        if view == "train":
            self.build_train_view()
        elif view == "detect":
            self.build_detect_view()
        elif view == "camera":
            self.build_camera_view()

    def clear_main(self):
        for child in self.main.winfo_children():
            child.destroy()
        for index in range(6):
            self.main.grid_rowconfigure(index, weight=0)
            self.main.grid_columnconfigure(index, weight=0)
        self.path_value_labels = {}
        self.main_lock_widgets = []

    def _clear_key_bindings(self):
        for sequence in ("<Left>", "<Right>", "<Return>", "<Escape>"):
            self.root.unbind(sequence)

    def _header(self, parent, title_key, subtitle_key):
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=24, pady=(22, 12))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text=self.t(title_key),
            font=("Segoe UI", 28, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            header,
            text=self.t(subtitle_key),
            font=("Segoe UI", 14),
            text_color=COLORS["muted"],
            anchor="w",
            wraplength=900,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

    def _panel(self, parent, **grid_options):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=10)
        frame.grid(**grid_options)
        return frame

    def _setup_resizable_columns(self, default_left, left_min=280, right_min=360):
        """配置 main 的三列布局（左控制栏 / 可拖动分隔条 / 右内容区）。

        拖动分隔条时实时改变左栏宽度，并保存到 self.left_panel_width，
        页面切换重建后沿用用户上次调整的宽度。
        """
        left = int(self.left_panel_width or default_left)
        self.main.grid_columnconfigure(0, weight=0, minsize=left)
        self.main.grid_columnconfigure(1, weight=0, minsize=6)
        self.main.grid_columnconfigure(2, weight=1)

        divider = tk.Frame(self.main, bg=self.c("line"), cursor="sb_h_double_arrow", width=6)
        divider.grid(row=1, column=1, sticky="nsew")
        divider.grid_propagate(False)

        state = {"start_x": 0, "start_w": left}

        def on_press(event):
            state["start_x"] = event.x_root
            state["start_w"] = self.main.grid_columnconfigure(0)["minsize"] or left
            divider.configure(bg=self.c("accent"))

        def on_drag(event):
            delta = event.x_root - state["start_x"]
            main_w = self.main.winfo_width()
            max_left = max(left_min, main_w - right_min - 6) if main_w > 0 else state["start_w"] + delta
            new_w = max(left_min, min(state["start_w"] + delta, max_left))
            self.main.grid_columnconfigure(0, minsize=int(new_w))
            self.left_panel_width = int(new_w)

        def on_release(_event):
            divider.configure(bg=self.c("line"))

        divider.bind("<ButtonPress-1>", on_press)
        divider.bind("<B1-Motion>", on_drag)
        divider.bind("<ButtonRelease-1>", on_release)
        divider.bind("<Enter>", lambda e: divider.configure(bg=self.c("accent_hover")))
        divider.bind("<Leave>", lambda e: divider.configure(bg=self.c("line")))

    def _form_entry(self, parent, row, label_key, placeholder_key, state_key):
        ctk.CTkLabel(
            parent,
            text=self.t(label_key),
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=14, pady=(12, 4))
        entry = ctk.CTkEntry(
            parent,
            height=40,
            corner_radius=8,
            placeholder_text=self.t(placeholder_key),
            font=("Segoe UI", 14),
            fg_color=COLORS["panel_alt"],
            text_color=COLORS["text"],
            border_color=COLORS["line"],
        )
        entry.insert(0, self.train_state.get(state_key, ""))
        entry.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.main_lock_widgets.append(entry)
        return entry

    def _path_selector(self, parent, row, label_key, attr_name, command, label_name):
        ctk.CTkLabel(
            parent,
            text=self.t(label_key),
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=14, pady=(12, 4))

        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 4))
        holder.grid_columnconfigure(0, weight=1)
        value = ctk.CTkLabel(
            holder,
            text=self._path_summary(getattr(self, attr_name, "")),
            anchor="w",
            justify="left",
            text_color=COLORS["muted"],
            fg_color=COLORS["panel_alt"],
            corner_radius=8,
            height=44,
            padx=10,
        )
        value.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        value.bind("<Button-1>", lambda _event, attr=attr_name: self.show_path_details(getattr(self, attr, "")))
        self.path_value_labels[label_name] = value
        select_button = ctk.CTkButton(
            holder,
            text=self.t("browse"),
            width=82,
            height=40,
            corner_radius=8,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            command=command,
        )
        select_button.grid(row=0, column=1, sticky="e")
        self.main_lock_widgets.append(select_button)

    def _path_summary(self, path):
        if not path:
            return self.t("not_selected")
        p = Path(path)
        parent = shorten_path(p.parent, 44)
        return f"{p.name}\n{parent}"

    def _refresh_path_label(self, label_name, path):
        label = self.path_value_labels.get(label_name)
        if label is not None and label.winfo_exists():
            label.configure(text=self._path_summary(path))

    def show_path_details(self, path):
        if self.busy:
            return
        if not path:
            return
        messagebox.showinfo(self.t("path_details"), path)

    def build_train_view(self):
        self.main.grid_rowconfigure(1, weight=1)
        self._setup_resizable_columns(460)
        self._header(self.main, "train_title", "train_subtitle")

        form = ctk.CTkScrollableFrame(self.main, fg_color=COLORS["bg"], corner_radius=0)
        form.grid(row=1, column=0, sticky="nsew", padx=(24, 10), pady=(0, 24))
        form.grid_columnconfigure(0, weight=1)

        row = 0
        self.project_name_entry = self._form_entry(form, row, "project_name", "project_placeholder", "project_name")
        row += 2

        ctk.CTkLabel(
            form,
            text=self.t("task_type"),
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=14, pady=(12, 4))
        self.task_menu = ctk.CTkOptionMenu(
            form,
            values=[self.t(f"task_{task}") for task in TASK_KEYS],
            height=40,
            corner_radius=8,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["blue_hover"],
            dropdown_fg_color=COLORS["panel_alt"],
            text_color=COLORS["text"],
            command=self.on_task_changed,
        )
        current_task = self.selected_task_var.get()
        self.task_menu.set(self.t(f"task_{current_task}"))
        self.task_menu.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.main_lock_widgets.append(self.task_menu)
        row += 2

        self._path_selector(form, row, "dataset_yaml", "data_yaml_path", self.select_data_yaml, "dataset_yaml")
        self.dataset_label = form.grid_slaves(row=row, column=0)[0]
        row += 2
        self._path_selector(form, row, "save_folder", "model_save_path", self.select_model_save_folder, "model_save")
        row += 2

        ctk.CTkLabel(
            form,
            text=self.t("yolo_model"),
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=14, pady=(12, 4))
        model_row = ctk.CTkFrame(form, fg_color="transparent")
        model_row.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 4))
        model_row.grid_columnconfigure(0, weight=1)
        self.selected_model_label = ctk.CTkLabel(
            model_row,
            text=self.selected_model_var.get(),
            anchor="w",
            text_color=COLORS["text"],
            fg_color=COLORS["panel_alt"],
            corner_radius=8,
            height=40,
            padx=10,
        )
        self.selected_model_label.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        model_select_button = ctk.CTkButton(
            model_row,
            text=self.t("open_model"),
            width=110,
            height=40,
            corner_radius=8,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            command=self.open_model_selection_window,
        )
        model_select_button.grid(row=0, column=1, sticky="e")
        self.main_lock_widgets.append(model_select_button)
        row += 2

        self.input_size_entry = self._form_entry(form, row, "input_size", "input_size_placeholder", "input_size")
        row += 2
        self.epochs_entry = self._form_entry(form, row, "epochs", "epochs_placeholder", "epochs")
        row += 2
        self.batch_size_entry = self._form_entry(form, row, "batch", "batch_placeholder", "batch_size")
        row += 2

        ctk.CTkLabel(
            form,
            text=self.t("train_device"),
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=14, pady=(12, 4))
        self.device_menu = ctk.CTkOptionMenu(
            form,
            values=[self.t(f"device_{option}") for option in DEVICE_OPTIONS],
            height=40,
            corner_radius=8,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["blue_hover"],
            dropdown_fg_color=COLORS["panel_alt"],
            text_color=COLORS["text"],
        )
        self.device_menu.set(self.t(f"device_{self.train_state['device']}"))
        self.device_menu.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.main_lock_widgets.append(self.device_menu)
        row += 2

        self.patience_entry = self._form_entry(form, row, "patience", "patience_placeholder", "patience")
        row += 2
        self.lr0_entry = self._form_entry(form, row, "lr0", "lr0_placeholder", "lr0")
        row += 2

        self.resume_switch = ctk.CTkCheckBox(
            form,
            text=self.t("resume_training"),
            variable=self.resume_var,
            font=("Segoe UI", 13),
            text_color=COLORS["text"],
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            corner_radius=6,
        )
        self.resume_switch.grid(row=row, column=0, sticky="w", padx=14, pady=(6, 2))
        self.main_lock_widgets.append(self.resume_switch)
        row += 1
        self.augment_switch = ctk.CTkCheckBox(
            form,
            text=self.t("augment"),
            variable=self.augment_var,
            font=("Segoe UI", 13),
            text_color=COLORS["text"],
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            corner_radius=6,
        )
        self.augment_switch.grid(row=row, column=0, sticky="w", padx=14, pady=(2, 6))
        self.main_lock_widgets.append(self.augment_switch)
        row += 1

        self.start_train_button = ctk.CTkButton(
            form,
            text=self.t("start_training"),
            height=48,
            corner_radius=8,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#071013",
            font=("Segoe UI", 16, "bold"),
            command=self.start_training,
        )
        self.start_train_button.grid(row=row, column=0, sticky="ew", padx=14, pady=(8, 8))

        self.train_button_progress_bar = ctk.CTkProgressBar(
            form,
            height=10,
            mode="indeterminate",
            indeterminate_speed=0.9,
            progress_color=COLORS["warning"],
        )
        self.train_button_progress_bar.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 20))
        self.train_button_progress_bar.set(0)

        log_panel = self._panel(self.main, row=1, column=2, sticky="nsew", padx=(10, 24), pady=(0, 24))
        log_panel.grid_rowconfigure(3, weight=1)
        log_panel.grid_columnconfigure(0, weight=1)

        device = self.get_device_summary()
        device_frame = ctk.CTkFrame(log_panel, fg_color=COLORS["panel_alt"], corner_radius=8)
        device_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        device_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(device_frame, text=self.t("device"), text_color=COLORS["muted"], anchor="w").grid(
            row=0, column=0, sticky="w", padx=12, pady=10
        )
        ctk.CTkLabel(
            device_frame,
            text=device,
            text_color=COLORS["accent"] if "GPU" in device else COLORS["warning"],
            font=("Segoe UI", 13, "bold"),
            anchor="e",
        ).grid(row=0, column=1, sticky="ew", padx=12, pady=10)

        progress_frame = ctk.CTkFrame(log_panel, fg_color="transparent")
        progress_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(2, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        self.training_status_label = ctk.CTkLabel(
            progress_frame,
            text=self.t("progress_idle"),
            text_color=COLORS["muted"],
            anchor="w",
            font=("Segoe UI", 13, "bold"),
        )
        self.training_status_label.grid(row=0, column=0, sticky="ew")
        self.training_eta_label = ctk.CTkLabel(
            progress_frame,
            text=f"{self.t('eta')}: {self.t('eta_unknown')}",
            text_color=COLORS["muted"],
            anchor="e",
        )
        self.training_eta_label.grid(row=0, column=1, sticky="e")
        self.training_progress_bar = ctk.CTkProgressBar(progress_frame, height=12, progress_color=COLORS["accent"])
        self.training_progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.training_progress_bar.set(0)
        self.training_metrics_label = ctk.CTkLabel(
            progress_frame,
            text=self.t("metrics_idle"),
            text_color=COLORS["muted"],
            anchor="w",
            font=("Consolas", 12),
        )
        self.training_metrics_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        ctk.CTkLabel(
            log_panel,
            text=self.t("log_title"),
            text_color=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 6))
        self.train_log_text = ctk.CTkTextbox(
            log_panel,
            corner_radius=8,
            fg_color=COLORS["log_bg"],
            text_color=COLORS["log_text"],
            font=("Consolas", 12),
            wrap="word",
        )
        self.train_log_text.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 16))

    def get_device_summary(self):
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                cuda = torch.version.cuda or "unknown CUDA"
                return f"GPU: {name} / CUDA {cuda}"
            return "CPU: CUDA GPU not detected"
        except Exception as exc:
            return f"Unknown: {exc}"

    def capture_train_state(self):
        for attr, key in [
            ("project_name_entry", "project_name"),
            ("input_size_entry", "input_size"),
            ("epochs_entry", "epochs"),
            ("batch_size_entry", "batch_size"),
            ("patience_entry", "patience"),
            ("lr0_entry", "lr0"),
        ]:
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    if widget.winfo_exists():
                        self.train_state[key] = widget.get()
                except tk.TclError:
                    pass
        device_menu = getattr(self, "device_menu", None)
        if device_menu is not None:
            try:
                if device_menu.winfo_exists():
                    shown = device_menu.get()
                    self.train_state["device"] = next(
                        (option for option in DEVICE_OPTIONS if self.t(f"device_{option}") == shown), "auto"
                    )
            except tk.TclError:
                pass
        self.train_state["resume"] = bool(self.resume_var.get())
        self.train_state["augment"] = bool(self.augment_var.get())

    def capture_view_state(self):
        self.capture_train_state()
        widget = getattr(self, "detection_conf_entry", None)
        if widget is not None:
            try:
                if widget.winfo_exists():
                    self.detection_confidence = widget.get().strip() or "0.5"
            except tk.TclError:
                pass
        widget = getattr(self, "camera_conf_entry", None)
        if widget is not None:
            try:
                if widget.winfo_exists():
                    self.camera_confidence = widget.get().strip() or "0.5"
            except tk.TclError:
                pass

    def select_data_yaml(self):
        if self.busy:
            return
        if self.selected_task_var.get() == "classify":
            path = filedialog.askdirectory(title=self.t("dataset_folder"))
        else:
            path = filedialog.askopenfilename(
                title=self.t("dataset_yaml"),
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            )
        if path:
            self.data_yaml_path = normalize_path(path)
            self._refresh_path_label("dataset_yaml", self.data_yaml_path)

    def select_model_save_folder(self):
        if self.busy:
            return
        path = filedialog.askdirectory()
        if path:
            self.model_save_path = normalize_path(path)
            self._refresh_path_label("model_save", self.model_save_path)

    def on_task_changed(self, display_text):
        """切换任务时，同步切换可选模型、数据集选择方式并重置当前选择。"""
        task = next((key for key in TASK_KEYS if self.t(f"task_{key}") == display_text), "detect")
        self.selected_task_var.set(task)
        first_model = MODEL_OPTIONS_BY_TASK[task][0][0]
        self.selected_model_var.set(first_model)
        if hasattr(self, "selected_model_label") and self.selected_model_label.winfo_exists():
            self.selected_model_label.configure(text=first_model)

        # 分类任务选目录，其余任务选 yaml；切换后清掉旧路径，避免类型不匹配
        self.data_yaml_path = ""
        self._refresh_path_label("dataset_yaml", "")
        if hasattr(self, "dataset_label") and self.dataset_label.winfo_exists():
            self.dataset_label.configure(
                text=self.t("dataset_folder") if task == "classify" else self.t("dataset_yaml")
            )

    def open_model_selection_window(self):
        if self.busy:
            return
        window = ctk.CTkToplevel(self.root)
        window.title(self.t("select_yolo_model"))
        window.geometry("420x620")
        window.transient(self.root)
        window.grab_set()

        ctk.CTkLabel(
            window,
            text=self.t("select_yolo_model"),
            font=("Segoe UI", 20, "bold"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(18, 8))
        frame = ctk.CTkScrollableFrame(window)
        frame.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        frame.grid_columnconfigure(0, weight=1)

        current_task = self.selected_task_var.get()
        model_names = [name for name, _ in MODEL_OPTIONS_BY_TASK[current_task]]
        current_family = ""
        for row, model_name in enumerate(model_names):
            family = model_name.split("-", 1)[0]
            if family != current_family:
                current_family = family
                ctk.CTkLabel(
                    frame,
                    text=family,
                    text_color=COLORS["muted"],
                    font=("Segoe UI", 13, "bold"),
                    anchor="w",
                ).grid(row=row * 2, column=0, sticky="ew", padx=8, pady=(12, 4))
            button = ctk.CTkButton(
                frame,
                text=model_name,
                height=36,
                corner_radius=8,
                fg_color=COLORS["panel_alt"],
                hover_color=COLORS["line"],
                text_color=COLORS["text"],
                command=lambda name=model_name: self.select_model_from_dialog(name, window),
            )
            button.grid(row=row * 2 + 1, column=0, sticky="ew", padx=8, pady=3)

    def select_model_from_dialog(self, model_name, window):
        self.selected_model_var.set(model_name)
        if hasattr(self, "selected_model_label") and self.selected_model_label.winfo_exists():
            self.selected_model_label.configure(text=model_name)
        window.destroy()

    def parse_confidence(self, entry):
        try:
            confidence = float(entry.get().strip())
        except (AttributeError, ValueError):
            messagebox.showwarning(self.t("invalid_title"), self.t("invalid_confidence"))
            return None
        if confidence < 0.01 or confidence > 1.0:
            messagebox.showwarning(self.t("invalid_title"), self.t("invalid_confidence"))
            return None
        return confidence

    def get_windows_camera_names(self):
        if os.name != "nt":
            return []
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_PnPEntity | "
                "Where-Object { $_.PNPClass -in @('Camera','Image') -or "
                "$_.Name -match 'Camera|Webcam|USB Video' } | "
                "Select-Object -ExpandProperty Name"
            ),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except Exception:
            return []
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    def discover_camera_devices(self, max_index=8):
        try:
            import cv2
        except Exception:
            return []

        names = self.get_windows_camera_names()
        devices = []
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        old_log_level = None
        if hasattr(cv2, "getLogLevel") and hasattr(cv2, "setLogLevel"):
            try:
                old_log_level = cv2.getLogLevel()
                cv2.setLogLevel(0)
            except Exception:
                old_log_level = None
        try:
            for camera_id in range(max_index):
                opened = False
                for backend in backends:
                    cap = cv2.VideoCapture(camera_id, backend)
                    opened = cap.isOpened()
                    cap.release()
                    if opened:
                        break
                if not opened:
                    continue
                name = names[len(devices)] if len(devices) < len(names) else f"Camera {camera_id}"
                devices.append({"id": camera_id, "name": name})
        finally:
            if old_log_level is not None:
                try:
                    cv2.setLogLevel(old_log_level)
                except Exception:
                    pass
        return devices

    def camera_device_values(self):
        self.camera_device_labels = {}
        values = []
        for device in self.camera_devices:
            label = f"{device['id']}: {device['name']}"
            self.camera_device_labels[label] = device["id"]
            values.append(label)
        return values

    def refresh_camera_device_menu(self):
        if self.busy:
            return
        self.camera_devices = self.discover_camera_devices()
        values = self.camera_device_values()
        if not hasattr(self, "camera_device_menu") or not self.camera_device_menu.winfo_exists():
            return
        if values:
            self.camera_device_menu.configure(values=values, state="normal")
            self.camera_device_var.set(values[0])
        else:
            self.camera_device_menu.configure(values=[self.t("no_camera_device")], state="disabled")
            self.camera_device_var.set(self.t("no_camera_device"))

    def validate_training_settings(self):
        self.capture_train_state()
        project_name = self.train_state["project_name"].strip()
        input_size = self.train_state["input_size"].strip()
        epochs = self.train_state["epochs"].strip()
        batch_size = self.train_state["batch_size"].strip()
        patience = self.train_state["patience"].strip()
        lr0 = self.train_state["lr0"].strip()
        current_task = self.selected_task_var.get()
        selected_model_size = MODEL_NAME_TO_TYPE_BY_TASK[current_task].get(self.selected_model_var.get(), "")

        missing = []
        if not project_name:
            missing.append(self.t("project_name"))
        if not self.data_yaml_path:
            missing.append(self.t("dataset_yaml"))
        if not self.model_save_path:
            missing.append(self.t("save_folder"))
        if not selected_model_size:
            missing.append(self.t("yolo_model"))
        if not input_size:
            missing.append(self.t("input_size"))
        if not epochs:
            missing.append(self.t("epochs"))
        if not batch_size:
            missing.append(self.t("batch"))
        if not patience:
            missing.append(self.t("patience"))
        if not lr0:
            missing.append(self.t("lr0"))

        if missing:
            messagebox.showwarning(
                self.t("missing_title"),
                self.t("missing_prefix") + "\n\n" + "\n".join(f"- {item}" for item in missing),
            )
            return None

        if not re.fullmatch(r"[A-Za-z0-9_-]+", project_name):
            messagebox.showwarning(self.t("invalid_title"), self.t("invalid_project"))
            return None

        dataset_path = Path(self.data_yaml_path)
        if current_task == "classify":
            if not dataset_path.is_dir():
                messagebox.showwarning(self.t("invalid_title"), self.t("folder_not_found"))
                return None
        elif not dataset_path.is_file():
            messagebox.showwarning(self.t("invalid_title"), self.t("yaml_not_found"))
            return None

        try:
            input_size_int = int(input_size)
            epochs_int = int(epochs)
            batch_size_int = int(batch_size)
            patience_int = int(patience)
            lr0_float = float(lr0)
            if input_size_int <= 0 or epochs_int <= 0 or batch_size_int <= 0 or patience_int < 0 or lr0_float <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning(self.t("invalid_title"), self.t("invalid_number"))
            return None

        return {
            "project_name": project_name,
            "input_size": input_size_int,
            "epochs": epochs_int,
            "batch_size": batch_size_int,
            "model_type": selected_model_size,
            "task": current_task,
            "patience": patience_int,
            "lr0": lr0_float,
            "device": self.train_state["device"],
            "resume": bool(self.resume_var.get()),
            "augment": bool(self.augment_var.get()),
        }

    def start_training(self):
        settings = self.validate_training_settings()
        if settings is None:
            return

        self.set_busy(True)
        self.training_stop_requested = False
        self.training_started_at = time.time()
        self.start_train_button.configure(
            text=self.t("training_click_stop"),
            state="normal",
            fg_color=COLORS["warning"],
            hover_color=COLORS["warning_hover"],
            text_color="#111827",
            command=self.stop_training,
        )
        self.training_status_label.configure(text=self.t("progress_running"), text_color=COLORS["accent"])
        self.training_eta_label.configure(text=f"{self.t('eta')}: {self.t('eta_unknown')}")
        self.training_progress_bar.set(0)
        if hasattr(self, "training_metrics_label") and self.training_metrics_label.winfo_exists():
            self.training_metrics_label.configure(text=self.t("metrics_idle"))
        if hasattr(self, "train_button_progress_bar") and self.train_button_progress_bar.winfo_exists():
            self.train_button_progress_bar.start()
        self.clear_train_log()
        self.append_train_log(self.t("train_started"))

        cmd_args = [
            sys.executable,
            "-m",
            "src.train",
            settings["project_name"],
            self.data_yaml_path,
            self.model_save_path,
            settings["model_type"],
            str(settings["input_size"]),
            str(settings["epochs"]),
            str(settings["batch_size"]),
            settings["task"],
            str(settings["patience"]),
            str(settings["lr0"]),
            settings["device"],
            "true" if settings["resume"] else "false",
            "false" if not settings["augment"] else "true",
        ]
        env = os.environ.copy()
        env.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

        thread = threading.Thread(target=self._run_training_process, args=(cmd_args, env), daemon=True)
        thread.start()

    def _run_training_process(self, cmd_args, env):
        try:
            self.train_process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            if self.training_stop_requested and self.train_process.poll() is None:
                self.train_process.terminate()
            for line in iter(self.train_process.stdout.readline, ""):
                for part in line.replace("\r", "\n").splitlines():
                    self.queue.put(("train_log", part))
            return_code = self.train_process.wait()
            self.queue.put(("train_done", return_code))
        except Exception:
            self.queue.put(("train_error", traceback.format_exc()))
        finally:
            self.train_process = None

    def stop_training(self):
        self.training_stop_requested = True
        self.append_train_log(self.t("stopping_training"))
        if hasattr(self, "start_train_button") and self.start_train_button.winfo_exists():
            self.start_train_button.configure(
                text=self.t("stopping_training"),
                state="disabled",
                fg_color=COLORS["warning"],
                hover_color=COLORS["warning_hover"],
            )
        if self.train_process is not None and self.train_process.poll() is None:
            self.train_process.terminate()

    def clear_train_log(self):
        if hasattr(self, "train_log_text") and self.train_log_text.winfo_exists():
            self.train_log_text.delete("1.0", "end")

    def append_train_log(self, line):
        if not line:
            return
        if hasattr(self, "train_log_text") and self.train_log_text.winfo_exists():
            self.train_log_text.insert("end", line.rstrip() + "\n")
            self.train_log_text.yview_moveto(1.0)

    def handle_train_log(self, line):
        match = PROGRESS_RE.search(line)
        if match:
            epoch = int(match.group("epoch"))
            total = max(int(match.group("total")), 1)
            eta = float(match.group("eta"))
            elapsed = float(match.group("elapsed"))
            progress = min(epoch / total, 1.0)
            if hasattr(self, "training_progress_bar") and self.training_progress_bar.winfo_exists():
                self.training_progress_bar.set(progress)
            if hasattr(self, "training_status_label") and self.training_status_label.winfo_exists():
                self.training_status_label.configure(text=f"{self.t('progress_running')} {epoch}/{total}")
            if hasattr(self, "training_eta_label") and self.training_eta_label.winfo_exists():
                self.training_eta_label.configure(
                    text=f"{self.t('eta')}: {format_duration(eta, self.language_var.get())} / "
                    f"{self.t('elapsed')}: {format_duration(elapsed, self.language_var.get())}"
                )
            metrics_text = (
                f"{self.t('metric_loss')} {match.group('loss') or 'na'}  |  "
                f"{self.t('metric_map50')} {match.group('map50') or 'na'}  |  "
                f"{self.t('metric_map5095')} {match.group('map5095') or 'na'}  |  "
                f"{self.t('metric_acc')} {match.group('acc') or 'na'}"
            )
            if hasattr(self, "training_metrics_label") and self.training_metrics_label.winfo_exists():
                self.training_metrics_label.configure(text=metrics_text)
            self.append_train_log(
                f"Epoch {epoch}/{total} - {metrics_text} - ETA {format_duration(eta, self.language_var.get())}"
            )
            return
        if line.startswith("GUI_DEVICE") or line.startswith("GUI_ARTIFACT"):
            self.append_train_log(line)
            return
        self.append_train_log(line)

    def finish_training(self, return_code):
        self.set_busy(False)
        if hasattr(self, "train_button_progress_bar") and self.train_button_progress_bar.winfo_exists():
            self.train_button_progress_bar.stop()
            self.train_button_progress_bar.set(0)
        if hasattr(self, "start_train_button") and self.start_train_button.winfo_exists():
            self.start_train_button.configure(
                text=self.t("start_training"),
                state="normal",
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"],
                text_color="#071013",
                command=self.start_training,
            )
        if self.training_stop_requested:
            if hasattr(self, "training_status_label") and self.training_status_label.winfo_exists():
                self.training_status_label.configure(text=self.t("progress_stopped"), text_color=COLORS["warning"])
            self.append_train_log(self.t("train_stopped"))
            self.training_stop_requested = False
            messagebox.showinfo(self.t("info"), self.t("train_stopped"))
        elif return_code == 0:
            if hasattr(self, "training_progress_bar") and self.training_progress_bar.winfo_exists():
                self.training_progress_bar.set(1)
            if hasattr(self, "training_status_label") and self.training_status_label.winfo_exists():
                self.training_status_label.configure(text=self.t("progress_done"), text_color=COLORS["success"])
            self.append_train_log(self.t("train_done"))
            messagebox.showinfo(self.t("info"), self.t("train_done"))
        else:
            if hasattr(self, "training_status_label") and self.training_status_label.winfo_exists():
                self.training_status_label.configure(text=self.t("progress_failed"), text_color=COLORS["danger"])
            self.append_train_log(self.t("train_failed"))
            messagebox.showerror(self.t("error"), self.t("train_failed"))

    def build_detect_view(self):
        self.main.grid_rowconfigure(1, weight=1)
        self._setup_resizable_columns(360)
        self._header(self.main, "detect_title", "detect_subtitle")

        controls = ctk.CTkScrollableFrame(self.main, fg_color=COLORS["bg"], corner_radius=0)
        controls.grid(row=1, column=0, sticky="nsew", padx=(24, 10), pady=(0, 24))
        controls.grid_columnconfigure(0, weight=1)

        row = 0
        self._path_selector(controls, row, "media_folder", "detection_images_folder_path", self.select_detection_folder, "detect_folder")
        row += 2

        self.selected_files_label = ctk.CTkLabel(
            controls,
            text=self.t("no_files_selected"),
            text_color=COLORS["muted"],
            anchor="w",
            wraplength=320,
            justify="left",
        )
        self.selected_files_label.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 4))
        pick_files_button = ctk.CTkButton(
            controls,
            text=self.t("select_images"),
            height=40,
            corner_radius=8,
            fg_color=COLORS["panel_alt"],
            hover_color=COLORS["blue"],
            command=self.select_detection_images,
        )
        pick_files_button.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.main_lock_widgets.append(pick_files_button)
        row += 2

        self._path_selector(controls, row, "model_file", "detection_model_path", self.select_detection_model, "detect_model")
        row += 2

        ctk.CTkLabel(
            controls,
            text=self.t("confidence"),
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=14, pady=(12, 4))
        self.detection_conf_entry = ctk.CTkEntry(
            controls,
            height=40,
            corner_radius=8,
            placeholder_text=self.t("confidence_placeholder"),
            fg_color=COLORS["panel_alt"],
            text_color=COLORS["text"],
            border_color=COLORS["line"],
        )
        self.detection_conf_entry.insert(0, self.detection_confidence)
        self.detection_conf_entry.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.main_lock_widgets.append(self.detection_conf_entry)
        row += 2

        self.start_detection_button = ctk.CTkButton(
            controls,
            text=self.t("start_detection"),
            height=48,
            corner_radius=8,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#071013",
            font=("Segoe UI", 16, "bold"),
            command=self.start_image_detection,
        )
        self.start_detection_button.grid(row=row, column=0, sticky="ew", padx=14, pady=(16, 10))
        row += 1

        self.detection_progress_bar = ctk.CTkProgressBar(controls, mode="indeterminate", height=10)
        self.detection_progress_bar.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 18))
        row += 1

        nav = ctk.CTkFrame(controls, fg_color="transparent")
        nav.grid(row=row, column=0, sticky="ew", padx=14, pady=(4, 8))
        nav.grid_columnconfigure((0, 1), weight=1)
        prev_button = ctk.CTkButton(nav, text=self.t("previous"), command=self.show_prev_image, height=38)
        prev_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        next_button = ctk.CTkButton(nav, text=self.t("next"), command=self.show_next_image, height=38)
        next_button.grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )
        self.main_lock_widgets.extend([prev_button, next_button])
        row += 1
        self.image_index_label = ctk.CTkLabel(
            controls,
            text=self.t("left_right_hint"),
            text_color=COLORS["muted"],
            anchor="w",
        )
        self.image_index_label.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 12))
        row += 1

        preview = self._panel(self.main, row=1, column=2, sticky="nsew", padx=(10, 24), pady=(0, 24))
        preview.grid_rowconfigure(0, weight=1)
        preview.grid_rowconfigure(1, weight=0)
        preview.grid_rowconfigure(2, weight=0)
        preview.grid_columnconfigure(0, weight=1)
        self.image_label = ctk.CTkLabel(
            preview,
            text=self.t("preview"),
            text_color=COLORS["muted"],
            fg_color=COLORS["preview_bg"],
            corner_radius=8,
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))

        ctk.CTkLabel(
            preview,
            text=self.t("detect_log_title"),
            text_color=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 6))

        self.detection_log = ctk.CTkTextbox(
            preview,
            height=200,
            corner_radius=8,
            fg_color=COLORS["log_bg"],
            text_color=COLORS["log_text"],
            font=("Consolas", 12),
            wrap="word",
        )
        self.detection_log.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        self.root.bind("<Left>", lambda _event: self.show_prev_image())
        self.root.bind("<Right>", lambda _event: self.show_next_image())
        if self.image_paths:
            self.update_image_preview()

    def select_detection_folder(self):
        if self.busy:
            return
        path = filedialog.askdirectory()
        if path:
            self.detection_images_folder_path = normalize_path(path)
            self.detection_selected_files = []
            self._refresh_path_label("detect_folder", self.detection_images_folder_path)
            self._refresh_selected_files_label()

    def select_detection_images(self):
        if self.busy:
            return
        paths = filedialog.askopenfilenames(
            title=self.t("select_images"),
            filetypes=[("Image/Video files", "*.jpg *.jpeg *.png *.bmp *.webp *.gif *.tiff "
                                             "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                       ("All files", "*.*")],
        )
        if paths:
            self.detection_selected_files = [normalize_path(p) for p in paths]
            self.detection_images_folder_path = ""
            self._refresh_path_label("detect_folder", "")
            self._refresh_selected_files_label()

    def _refresh_selected_files_label(self):
        label = getattr(self, "selected_files_label", None)
        if label is None or not label.winfo_exists():
            return
        count = len(self.detection_selected_files)
        if count:
            label.configure(text=self.t("files_selected").format(count=count), text_color=COLORS["text"])
        else:
            label.configure(text=self.t("no_files_selected"), text_color=COLORS["muted"])

    def select_detection_model(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(filetypes=[(self.t("file_model_filter"), "*.pt"), (self.t("all_models"), "*.*")])
        if path:
            self.detection_model_path = normalize_path(path)
            self._refresh_path_label("detect_model", self.detection_model_path)

    def start_image_detection(self):
        missing = []
        if not self.detection_images_folder_path and not self.detection_selected_files:
            missing.append(self.t("media_hint"))
        if not self.detection_model_path:
            missing.append(self.t("model_file"))
        if missing:
            messagebox.showwarning(
                self.t("missing_title"),
                self.t("missing_prefix") + "\n\n" + "\n".join(f"- {item}" for item in missing),
            )
            return
        confidence = self.parse_confidence(self.detection_conf_entry)
        if confidence is None:
            return
        self.detection_confidence = self.detection_conf_entry.get().strip()

        self.set_busy(True)
        self.image_paths = []
        self.current_image_index = 0
        if hasattr(self, "start_detection_button") and self.start_detection_button.winfo_exists():
            self.start_detection_button.configure(
                text=self.t("detect_running"),
                state="disabled",
                fg_color=COLORS["warning"],
                hover_color=COLORS["warning_hover"],
                text_color="#111827",
            )
        self.detection_progress_bar.start()
        self.clear_detection_log()
        self.append_detection_log(self.t("detect_running"))

        thread = threading.Thread(target=self._run_image_detection, args=(confidence,), daemon=True)
        thread.start()

    def _run_image_detection(self, confidence):
        try:
            from src.detect import detect_images

            def progress(message):
                self.queue.put(("detect_log", message))

            def done(results_dir):
                self.queue.put(("detect_done", results_dir))

            detect_images(
                self.detection_images_folder_path,
                self.detection_model_path,
                callback=done,
                progress_callback=progress,
                conf_threshold=confidence,
                selected_files=self.detection_selected_files or None,
            )
        except Exception:
            self.queue.put(("detect_error", traceback.format_exc()))

    def clear_detection_log(self):
        if hasattr(self, "detection_log") and self.detection_log.winfo_exists():
            self.detection_log.delete("1.0", "end")

    def append_detection_log(self, line):
        if hasattr(self, "detection_log") and self.detection_log.winfo_exists():
            self.detection_log.insert("end", str(line).rstrip() + "\n")
            self.detection_log.yview_moveto(1.0)

    def finish_detection(self, results_dir):
        self.set_busy(False)
        if hasattr(self, "start_detection_button") and self.start_detection_button.winfo_exists():
            self.start_detection_button.configure(
                text=self.t("start_detection"),
                state="normal",
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"],
                text_color="#071013",
            )
        if hasattr(self, "detection_progress_bar") and self.detection_progress_bar.winfo_exists():
            self.detection_progress_bar.stop()

        self.image_paths = self.find_result_images(results_dir)
        self.current_image_index = 0
        self.append_detection_log(self.t("detect_done"))
        if self.image_paths:
            self.update_image_preview()
        else:
            self.append_detection_log(self.t("no_results"))
            if hasattr(self, "image_label") and self.image_label.winfo_exists():
                self.image_label.configure(text=self.t("no_results"), image=None)
        messagebox.showinfo(self.t("info"), self.t("detect_done"))

    def find_result_images(self, results_dir):
        root = Path(results_dir)
        if not root.exists():
            return []
        return sorted(str(path) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

    def update_image_preview(self):
        if not self.image_paths or not hasattr(self, "image_label") or not self.image_label.winfo_exists():
            return

        label_width = max(self.image_label.winfo_width() - 24, 1)
        label_height = max(self.image_label.winfo_height() - 24, 1)
        if label_width <= 1 or label_height <= 1:
            self.root.after(100, self.update_image_preview)
            return

        image_path = self.image_paths[self.current_image_index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_image, text="")
        if hasattr(self, "image_index_label") and self.image_index_label.winfo_exists():
            self.image_index_label.configure(
                text=f"{self.current_image_index + 1}/{len(self.image_paths)}   {self.t('left_right_hint')}"
            )

    def show_next_image(self):
        if self.busy:
            return
        if not self.image_paths:
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.update_image_preview()

    def show_prev_image(self):
        if self.busy:
            return
        if not self.image_paths:
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.update_image_preview()

    def build_camera_view(self):
        self.main.grid_rowconfigure(1, weight=1)
        self._setup_resizable_columns(360)
        self._header(self.main, "camera_title", "camera_subtitle")

        controls = ctk.CTkScrollableFrame(self.main, fg_color=COLORS["bg"], corner_radius=0)
        controls.grid(row=1, column=0, sticky="nsew", padx=(24, 10), pady=(0, 24))
        controls.grid_columnconfigure(0, weight=1)

        row = 0
        self._path_selector(controls, row, "model_file", "detection_model_path", self.select_camera_model, "camera_model")
        row += 2
        self._path_selector(controls, row, "camera_save", "detection_save_dir", self.select_camera_save_folder, "camera_save")
        row += 2
        ctk.CTkLabel(
            controls,
            text=self.t("camera_device"),
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=14, pady=(12, 4))
        device_row = ctk.CTkFrame(controls, fg_color="transparent")
        device_row.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 12))
        device_row.grid_columnconfigure(0, weight=1)
        self.camera_devices = self.discover_camera_devices()
        camera_values = self.camera_device_values()
        if camera_values:
            self.camera_device_var.set(camera_values[0])
        else:
            camera_values = [self.t("no_camera_device")]
            self.camera_device_var.set(camera_values[0])
        self.camera_device_menu = ctk.CTkOptionMenu(
            device_row,
            variable=self.camera_device_var,
            values=camera_values,
            height=40,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["blue"],
            button_hover_color=COLORS["blue_hover"],
            text_color=COLORS["text"],
            state="normal" if self.camera_device_labels else "disabled",
        )
        self.camera_device_menu.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        refresh_button = ctk.CTkButton(
            device_row,
            text=self.t("camera_refresh"),
            width=86,
            height=40,
            fg_color=COLORS["blue"],
            hover_color=COLORS["blue_hover"],
            command=self.refresh_camera_device_menu,
        )
        refresh_button.grid(row=0, column=1, sticky="e")
        self.main_lock_widgets.extend([self.camera_device_menu, refresh_button])
        row += 2

        ctk.CTkLabel(
            controls,
            text=self.t("confidence"),
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=14, pady=(12, 4))
        self.camera_conf_entry = ctk.CTkEntry(
            controls,
            height=40,
            corner_radius=8,
            placeholder_text=self.t("confidence_placeholder"),
            fg_color=COLORS["panel_alt"],
            text_color=COLORS["text"],
            border_color=COLORS["line"],
        )
        self.camera_conf_entry.insert(0, self.camera_confidence)
        self.camera_conf_entry.grid(row=row + 1, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.main_lock_widgets.append(self.camera_conf_entry)
        row += 2

        self.camera_button = ctk.CTkButton(
            controls,
            text=self.t("start"),
            height=48,
            corner_radius=8,
            fg_color=COLORS["success"],
            hover_color="#16a34a",
            font=("Segoe UI", 16, "bold"),
            command=self.start_camera_detection,
        )
        self.camera_button.grid(row=row, column=0, sticky="ew", padx=14, pady=(8, 12))
        row += 1

        self.camera_status_label = ctk.CTkLabel(
            controls,
            text=f"{self.t('enter_hint')}   {self.t('esc_hint')}",
            text_color=COLORS["muted"],
            anchor="w",
        )
        self.camera_status_label.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 16))

        preview = self._panel(self.main, row=1, column=2, sticky="nsew", padx=(10, 24), pady=(0, 24))
        preview.grid_rowconfigure(0, weight=1)
        preview.grid_columnconfigure(0, weight=1)
        self.camera_label = ctk.CTkLabel(
            preview,
            text=self.t("camera_title"),
            text_color=COLORS["muted"],
            fg_color=COLORS["preview_bg"],
            corner_radius=8,
        )
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self.root.bind("<Return>", lambda _event: self.save_camera_frame())
        self.root.bind("<Escape>", lambda _event: self.stop_camera_detection())

    def select_camera_model(self):
        if self.busy:
            return
        self.select_detection_model()
        self._refresh_path_label("camera_model", self.detection_model_path)

    def select_camera_save_folder(self):
        if self.busy:
            return
        path = filedialog.askdirectory()
        if path:
            self.detection_save_dir = normalize_path(path)
            self._refresh_path_label("camera_save", self.detection_save_dir)
            if self.camera_detection is not None:
                self.camera_detection.set_save_directory(self.detection_save_dir)

    def start_camera_detection(self):
        missing = []
        if not self.detection_model_path:
            missing.append(self.t("model_file"))
        if not self.detection_save_dir:
            missing.append(self.t("camera_save"))
        if missing:
            messagebox.showwarning(
                self.t("missing_title"),
                self.t("missing_prefix") + "\n\n" + "\n".join(f"- {item}" for item in missing),
            )
            return
        camera_label = self.camera_device_var.get()
        camera_id = self.camera_device_labels.get(camera_label)
        if camera_id is None:
            messagebox.showwarning(self.t("missing_title"), self.t("no_camera_device"))
            return
        confidence = self.parse_confidence(self.camera_conf_entry)
        if confidence is None:
            return
        self.camera_confidence = self.camera_conf_entry.get().strip()

        self.set_busy(True)
        self.camera_button.configure(state="disabled")
        self.camera_status_label.configure(text=self.t("camera_running"), text_color=COLORS["accent"])
        threading.Thread(target=self._load_camera, args=(camera_id, confidence), daemon=True).start()

    def _load_camera(self, camera_id, confidence):
        try:
            from src.camera import CameraDetection

            camera = CameraDetection(self.detection_model_path, conf_threshold=confidence)
            camera.set_save_directory(self.detection_save_dir)
            camera.start_camera(camera_id)
            self.queue.put(("camera_started", camera))
        except Exception:
            self.queue.put(("camera_error", traceback.format_exc()))

    def finish_camera_started(self, camera):
        self.camera_detection = camera
        self.camera_detection.show_camera_stream(self.camera_label)
        self.camera_button.configure(
            state="normal",
            text=self.t("stop"),
            fg_color=COLORS["danger"],
            hover_color=COLORS["danger_hover"],
            command=self.stop_camera_detection,
        )
        self.camera_status_label.configure(text=self.t("camera_running"), text_color=COLORS["accent"])

    def stop_camera_detection(self):
        if self.camera_detection is not None:
            self.camera_detection.stop()
            self.camera_detection = None
        self.set_busy(False)
        if hasattr(self, "camera_button") and self.camera_button.winfo_exists():
            self.camera_button.configure(
                state="normal",
                text=self.t("start"),
                fg_color=COLORS["success"],
                hover_color="#16a34a",
                command=self.start_camera_detection,
            )
        if hasattr(self, "camera_status_label") and self.camera_status_label.winfo_exists():
            self.camera_status_label.configure(text=self.t("camera_stopped"), text_color=COLORS["muted"])

    def save_camera_frame(self):
        if self.camera_detection is None:
            return
        try:
            result = self.camera_detection.capture_frame()
        except Exception:
            result = None
        if result:
            self.camera_status_label.configure(text=self.t("capture_done"), text_color=COLORS["accent"])
        else:
            self.camera_status_label.configure(text=self.t("capture_failed"), text_color=COLORS["warning"])
            messagebox.showwarning(self.t("error"), self.t("capture_failed"))

    def process_queue(self):
        try:
            while True:
                event, payload = self.queue.get_nowait()
                if event == "train_log":
                    self.handle_train_log(payload)
                elif event == "train_done":
                    self.finish_training(payload)
                elif event == "train_error":
                    self.append_train_log(payload)
                    self.finish_training(1)
                elif event == "detect_log":
                    self.append_detection_log(payload)
                elif event == "detect_done":
                    self.finish_detection(payload)
                elif event == "detect_error":
                    self.set_busy(False)
                    if hasattr(self, "start_detection_button") and self.start_detection_button.winfo_exists():
                        self.start_detection_button.configure(
                            text=self.t("start_detection"),
                            state="normal",
                            fg_color=COLORS["accent"],
                            hover_color=COLORS["accent_hover"],
                            text_color="#071013",
                        )
                    if hasattr(self, "detection_progress_bar") and self.detection_progress_bar.winfo_exists():
                        self.detection_progress_bar.stop()
                    self.append_detection_log(payload)
                    messagebox.showerror(self.t("error"), self.t("detect_failed"))
                elif event == "camera_started":
                    self.finish_camera_started(payload)
                elif event == "camera_error":
                    self.stop_camera_detection()
                    if hasattr(self, "camera_status_label") and self.camera_status_label.winfo_exists():
                        self.camera_status_label.configure(text=payload, text_color=COLORS["danger"])
                    messagebox.showerror(self.t("error"), payload)
        except Empty:
            pass
        self.root.after(100, self.process_queue)

    def on_close(self):
        if self.train_process is not None:
            messagebox.showinfo(self.t("info"), self.t("blocked"))
            return
        if self.camera_detection is not None:
            self.camera_detection.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = YoloGuiApp()
    app.run()
