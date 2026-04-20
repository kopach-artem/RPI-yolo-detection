from __future__ import annotations

import cv2

from app.types import Detection


def annotate_frame(
    frame,
    detections: list[Detection],
    alert_labels: set[str],
    moving_indexes: set[int] | None = None,
    confirm_min_frames: int | None = None,
):
    annotated = frame.copy()
    moving_indexes = moving_indexes or set()

    for idx, det in enumerate(detections):
        label = det.label
        conf = det.confidence
        x1, y1, x2, y2 = det.box_xyxy

        color = (0, 255, 0)
        text_suffix = ""

        if label in alert_labels:
            color = (0, 165, 255)

            if idx in moving_indexes:
                if det.is_confirmed:
                    color = (0, 0, 255)
                    text_suffix = " MOVING CONFIRMED"
                elif confirm_min_frames and confirm_min_frames > 1:
                    color = (0, 140, 255)
                    text_suffix = f" MOVING {det.confirm_hits}/{confirm_min_frames}"
                else:
                    color = (0, 0, 255)
                    text_suffix = " MOVING"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}{text_suffix}"
        cv2.putText(
            annotated,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated


def draw_status_overlay(
    frame,
    fps: float,
    detections_count: int,
    motion_found: bool,
    motion_area: int,
    model_name: str,
    current_stride: int,
    alert_counts: dict[str, int] | None = None,
    moving_alert_counts: dict[str, int] | None = None,
    confirmed_alert_counts: dict[str, int] | None = None,
):
    annotated = frame

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f} | obj: {detections_count} | motion: {int(motion_found)} | area: {int(motion_area)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        annotated,
        f"model: {model_name} | stride: {current_stride}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    if alert_counts:
        seen_text = ", ".join(f"{k}:{v}" for k, v in sorted(alert_counts.items()))
        cv2.putText(
            annotated,
            f"SEEN: {seen_text}",
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 165, 255),
            2,
        )

    if moving_alert_counts:
        moving_text = ", ".join(f"{k}:{v}" for k, v in sorted(moving_alert_counts.items()))
        cv2.putText(
            annotated,
            f"MOVING CANDIDATES: {moving_text}",
            (10, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 140, 255),
            2,
        )

    if confirmed_alert_counts:
        confirmed_text = ", ".join(
            f"{k}:{v}" for k, v in sorted(confirmed_alert_counts.items())
        )
        cv2.putText(
            annotated,
            f"CONFIRMED ALERT: {confirmed_text}",
            (10, 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 255),
            2,
        )

    return annotated


def encode_jpeg(frame, jpeg_quality: int) -> bytes | None:
    ok_jpg, jpg = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok_jpg:
        return None
    return jpg.tobytes()


def build_counts(detections: list[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts


def filter_alert_detections(
    detections: list[Detection],
    alert_labels: set[str],
) -> list[Detection]:
    return [det for det in detections if det.label in alert_labels]


def filter_moving_detections(
    detections: list[Detection],
    alert_labels: set[str],
) -> list[Detection]:
    return [
        det
        for det in detections
        if det.label in alert_labels and det.is_moving
    ]


def filter_confirmed_moving_detections(
    detections: list[Detection],
    alert_labels: set[str],
) -> list[Detection]:
    return [
        det
        for det in detections
        if det.label in alert_labels and det.is_moving and det.is_confirmed
    ]
