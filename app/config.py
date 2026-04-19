from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    stream_url: str
    model_name: str

    save_every: float
    conf: float
    imgsz: int
    http_port: int
    jpeg_quality: int

    idle_frame_stride: int
    active_frame_stride: int
    detection_stale_sec: float

    telegram_bot_token: str
    telegram_chat_id: str
    telegram_cooldown: float

    alert_labels: set[str]

    motion_gate: bool
    motion_min_area: int
    motion_warmup_frames: int
    motion_box_pad: int
    motion_box_min_pixels: int
    motion_box_min_ratio: float

    shots_dir: Path
    json_dir: Path
    alerts_dir: Path


def _read_alert_labels(raw: str) -> set[str]:
    return {
        label.strip().lower()
        for label in raw.split(",")
        if label.strip()
    }


def load_config() -> AppConfig:
    data_root = Path(os.getenv("DATA_ROOT", "/work/data"))

    cfg = AppConfig(
        stream_url=os.getenv("STREAM_URL", "http://192.168.0.158:4747/video"),
        model_name=os.getenv("MODEL_NAME", "yolo26s.pt"),

        save_every=float(os.getenv("SAVE_EVERY", "5")),
        conf=float(os.getenv("CONF", "0.25")),
        imgsz=int(os.getenv("IMGSZ", "320")),
        http_port=int(os.getenv("HTTP_PORT", "8080")),
        jpeg_quality=int(os.getenv("JPEG_QUALITY", "65")),

        idle_frame_stride=int(os.getenv("IDLE_FRAME_STRIDE", os.getenv("FRAME_STRIDE", "3"))),
        active_frame_stride=int(os.getenv("ACTIVE_FRAME_STRIDE", "1")),
        detection_stale_sec=float(os.getenv("DETECTION_STALE_SEC", "1.5")),

        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        telegram_cooldown=float(os.getenv("TELEGRAM_COOLDOWN", "10")),

        alert_labels=_read_alert_labels(os.getenv("ALERT_LABELS", "dog,person")),

        motion_gate=os.getenv("MOTION_GATE", "1") == "1",
        motion_min_area=int(os.getenv("MOTION_MIN_AREA", "5000")),
        motion_warmup_frames=int(os.getenv("MOTION_WARMUP_FRAMES", "30")),
        motion_box_pad=int(os.getenv("MOTION_BOX_PAD", "12")),
        motion_box_min_pixels=int(os.getenv("MOTION_BOX_MIN_PIXELS", "80")),
        motion_box_min_ratio=float(os.getenv("MOTION_BOX_MIN_RATIO", "0.01")),

        shots_dir=data_root / "shots",
        json_dir=data_root / "json",
        alerts_dir=data_root / "alerts",
    )

    validate_config(cfg)
    ensure_runtime_dirs(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    if not cfg.stream_url:
        raise ValueError("STREAM_URL must not be empty")

    if not cfg.model_name:
        raise ValueError("MODEL_NAME must not be empty")

    if cfg.idle_frame_stride < 1:
        raise ValueError("IDLE_FRAME_STRIDE must be >= 1")

    if cfg.active_frame_stride < 1:
        raise ValueError("ACTIVE_FRAME_STRIDE must be >= 1")

    if cfg.detection_stale_sec < 0:
        raise ValueError("DETECTION_STALE_SEC must be >= 0")

    if not (0.0 < cfg.conf <= 1.0):
        raise ValueError("CONF must be in (0, 1]")

    if cfg.imgsz <= 0:
        raise ValueError("IMGSZ must be > 0")

    if cfg.http_port <= 0 or cfg.http_port > 65535:
        raise ValueError("HTTP_PORT must be in 1..65535")

    if cfg.jpeg_quality < 1 or cfg.jpeg_quality > 100:
        raise ValueError("JPEG_QUALITY must be in 1..100")

    if cfg.save_every < 0:
        raise ValueError("SAVE_EVERY must be >= 0")

    if cfg.telegram_cooldown < 0:
        raise ValueError("TELEGRAM_COOLDOWN must be >= 0")

    if cfg.motion_min_area < 0:
        raise ValueError("MOTION_MIN_AREA must be >= 0")

    if cfg.motion_warmup_frames < 0:
        raise ValueError("MOTION_WARMUP_FRAMES must be >= 0")

    if cfg.motion_box_pad < 0:
        raise ValueError("MOTION_BOX_PAD must be >= 0")

    if cfg.motion_box_min_pixels < 0:
        raise ValueError("MOTION_BOX_MIN_PIXELS must be >= 0")

    if cfg.motion_box_min_ratio < 0:
        raise ValueError("MOTION_BOX_MIN_RATIO must be >= 0")


def ensure_runtime_dirs(cfg: AppConfig) -> None:
    cfg.shots_dir.mkdir(parents=True, exist_ok=True)
    cfg.json_dir.mkdir(parents=True, exist_ok=True)
    cfg.alerts_dir.mkdir(parents=True, exist_ok=True)
