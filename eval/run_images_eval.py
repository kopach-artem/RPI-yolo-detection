from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from app.config import load_config, apply_run_mode
from app.detector import YoloDetector

CORE_CLASSES = {"person", "dog"}
WIDE_CLASSES = {
    "person",
    "dog",
    "cat",
    "car",
    "chair",
    "cup",
    "bottle",
    "cell phone",
    "laptop",
    "keyboard",
    "backpack",
    "book",
}

IMAGES_ROOT = Path("eval_data/images")
OUTPUT_ROOT = Path("eval_runs/images")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["core", "wide"],
        required=True,
        help="Which prepared subset to evaluate",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated run modes. Example: 0,1,2,3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images to process",
    )
    return parser.parse_args()


def default_modes_for_subset(subset: str) -> list[int]:
    if subset == "core":
        return [0, 1, 2, 3]
    if subset == "wide":
        return [0, 1, 3]
    raise ValueError(f"Unsupported subset: {subset}")


def allowed_classes_for_subset(subset: str) -> set[str]:
    if subset == "core":
        return CORE_CLASSES
    if subset == "wide":
        return WIDE_CLASSES
    raise ValueError(f"Unsupported subset: {subset}")


def parse_modes(modes_arg: str | None, subset: str) -> list[int]:
    if not modes_arg:
        return default_modes_for_subset(subset)

    modes = []
    for part in modes_arg.split(","):
        part = part.strip()
        if not part:
            continue
        mode = int(part)
        modes.append(mode)

    return modes


def list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(files)


def detections_to_jsonable(detections) -> list[dict]:
    out = []
    for det in detections:
        out.append(
            {
                "label": det.label,
                "confidence": round(float(det.confidence), 6),
                "box_xyxy": [int(v) for v in det.box_xyxy],
            }
        )
    return out


def run_subset_mode(subset: str, run_mode: int, image_paths: list[Path]) -> dict:
    cfg = load_config()
    cfg = apply_run_mode(cfg, run_mode)

    detector = YoloDetector(cfg)
    allowed_classes = allowed_classes_for_subset(subset)

    results = {
        "subset": subset,
        "run_mode": run_mode,
        "model_name": cfg.model_name,
        "images_count": len(image_paths),
        "predictions": [],
    }

    print(f"\n=== Running subset={subset} mode={run_mode} model={cfg.model_name} ===")

    for idx, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        detections = detector.predict(frame)
        detections = [det for det in detections if det.label in allowed_classes]

        item = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "detections": detections_to_jsonable(detections),
        }
        results["predictions"].append(item)

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)}")

    return results


def main():
    args = parse_args()

    subset = args.subset
    modes = parse_modes(args.modes, subset)

    images_dir = IMAGES_ROOT / subset
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    image_paths = list_images(images_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    out_dir = OUTPUT_ROOT / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subset: {subset}")
    print(f"Images dir: {images_dir}")
    print(f"Images count: {len(image_paths)}")
    print(f"Modes: {modes}")

    for run_mode in modes:
        results = run_subset_mode(subset, run_mode, image_paths)

        out_path = out_dir / f"mode_{run_mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
