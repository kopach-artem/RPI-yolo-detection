from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2

from app.config import AppConfig
from app.types import Detection


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_image(image_path: Path, frame) -> bool:
    try:
        ok = cv2.imwrite(str(image_path), frame)
    except Exception as exc:
        print("Image save exception:", exc)
        return False

    if not ok:
        print("Image save failed:", image_path)
        return False

    return True


def detections_to_dicts(detections: list[Detection]) -> list[dict]:
    return [det.to_dict() for det in detections]


def build_payload(
    cfg: AppConfig,
    ts: str,
    detections: list[Detection],
    alert_detections: list[Detection],
    moving_alert_detections: list[Detection],
    motion_found: bool,
    motion_area: int,
) -> dict:
    return {
        "timestamp": ts,
        "stream_url": cfg.stream_url,
        "model_name": cfg.model_name,
        "alert_labels": sorted(cfg.alert_labels),
        "save_every_sec": cfg.save_every,
        "idle_frame_stride": cfg.idle_frame_stride,
        "active_frame_stride": cfg.active_frame_stride,
        "motion_gate": cfg.motion_gate,
        "motion_min_area": cfg.motion_min_area,
        "motion_box_pad": cfg.motion_box_pad,
        "motion_box_min_pixels": cfg.motion_box_min_pixels,
        "motion_box_min_ratio": cfg.motion_box_min_ratio,
        "motion_found": bool(motion_found),
        "motion_area": int(motion_area),
        "detections_count": len(detections),
        "alert_counts": build_counts(alert_detections),
        "moving_alert_counts": build_counts(moving_alert_detections),
        "detections": detections_to_dicts(detections),
    }


def save_payload(json_path: Path, payload: dict) -> bool:
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print("JSON save exception:", exc)
        return False
    return True


def save_event_artifacts(
    cfg: AppConfig,
    ts: str,
    annotated_frame,
    detections: list[Detection],
    alert_detections: list[Detection],
    moving_alert_detections: list[Detection],
    motion_found: bool,
    motion_area: int,
) -> tuple[Path, Path]:
    jpg_path = cfg.shots_dir / f"{ts}.jpg"
    json_path = cfg.json_dir / f"{ts}.json"

    save_image(jpg_path, annotated_frame)

    payload = build_payload(
        cfg=cfg,
        ts=ts,
        detections=detections,
        alert_detections=alert_detections,
        moving_alert_detections=moving_alert_detections,
        motion_found=motion_found,
        motion_area=motion_area,
    )
    save_payload(json_path, payload)

    return jpg_path, json_path


def save_alert_snapshot(
    cfg: AppConfig,
    ts: str,
    annotated_frame,
    moving_alert_detections: list[Detection],
) -> Path:
    labels = sorted({det.label for det in moving_alert_detections})
    labels_part = "_".join(labels) if labels else "unknown"

    alert_path = cfg.alerts_dir / f"{ts}_{labels_part}.jpg"
    save_image(alert_path, annotated_frame)
    return alert_path


def build_counts(detections: list[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts
