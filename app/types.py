from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: list[int]

    is_moving: bool = False
    motion_pixels: int = 0
    motion_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box_xyxy": [int(v) for v in self.box_xyxy],
            "is_moving": bool(self.is_moving),
            "motion_pixels": int(self.motion_pixels),
            "motion_ratio": round(float(self.motion_ratio), 4),
        }


@dataclass(slots=True)
class MotionResult:
    found: bool
    area: int
    mask: Any = None


@dataclass(slots=True)
class FrameState:
    frame_idx: int = 0
    fps: float = 0.0
    last_time: float = 0.0
    last_infer_time: float = 0.0
    last_save: float = 0.0
    last_alert: float = 0.0
    warmup_count: int = 0


@dataclass(slots=True)
class DetectionCache:
    detections: list[Detection] = field(default_factory=list)
    updated_at: float = 0.0
