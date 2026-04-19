from __future__ import annotations

import cv2

from app.types import Detection, MotionResult


def create_motion_subtractor(
    history: int = 300,
    var_threshold: int = 25,
    detect_shadows: bool = False,
):
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )


def detect_motion(frame, subtractor, min_area: int = 5000) -> MotionResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    fgmask = subtractor.apply(gray)

    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    max_area = 0
    motion_found = False

    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area > max_area:
            max_area = area
        if area >= min_area:
            motion_found = True

    return MotionResult(
        found=motion_found,
        area=max_area,
        mask=thresh,
    )


def clamp_box(
    box: list[int] | tuple[int, int, int, int],
    width: int,
    height: int,
    pad: int = 0,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    return x1, y1, x2, y2


def box_has_motion(
    box: list[int] | tuple[int, int, int, int],
    motion_mask,
    width: int,
    height: int,
    pad: int = 12,
    min_pixels: int = 80,
    min_ratio: float = 0.01,
) -> tuple[bool, int, float]:
    if motion_mask is None:
        return False, 0, 0.0

    x1, y1, x2, y2 = clamp_box(box, width, height, pad=pad)

    if x2 <= x1 or y2 <= y1:
        return False, 0, 0.0

    roi = motion_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0, 0.0

    motion_pixels = int(cv2.countNonZero(roi))
    box_area = int((x2 - x1) * (y2 - y1))
    motion_ratio = (motion_pixels / box_area) if box_area > 0 else 0.0

    is_moving = (
        motion_pixels >= min_pixels
        and motion_ratio >= min_ratio
    )

    return is_moving, motion_pixels, motion_ratio


def enrich_detections_with_motion(
    detections: list[Detection],
    motion_mask,
    frame_width: int,
    frame_height: int,
    pad: int = 12,
    min_pixels: int = 80,
    min_ratio: float = 0.01,
) -> tuple[list[Detection], set[int]]:
    moving_indexes: set[int] = set()

    for idx, det in enumerate(detections):
        is_moving, motion_pixels, motion_ratio = box_has_motion(
            det.box_xyxy,
            motion_mask,
            frame_width,
            frame_height,
            pad=pad,
            min_pixels=min_pixels,
            min_ratio=min_ratio,
        )

        det.is_moving = is_moving
        det.motion_pixels = motion_pixels
        det.motion_ratio = round(motion_ratio, 4)

        if is_moving:
            moving_indexes.add(idx)

    return detections, moving_indexes
