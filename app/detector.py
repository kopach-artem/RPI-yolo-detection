from __future__ import annotations

from ultralytics import YOLO

from app.config import AppConfig
from app.types import Detection


class YoloDetector:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_name)

    def predict(self, frame) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            device="cpu",
            verbose=False,
        )

        if not results:
            return []

        return self._result_to_detections(results[0])

    def _result_to_detections(self, result) -> list[Detection]:
        detections: list[Detection] = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xyxy = result.boxes.xyxy.cpu().tolist()
        cls_ids = result.boxes.cls.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        for box, cls_id, conf in zip(xyxy, cls_ids, confs):
            label = result.names[int(cls_id)].lower()
            x1, y1, x2, y2 = [int(v) for v in box]

            detections.append(
                Detection(
                    label=label,
                    confidence=float(conf),
                    box_xyxy=[x1, y1, x2, y2],
                )
            )

        return detections

    def class_names(self) -> dict[int, str]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(name) for i, name in enumerate(names)}
