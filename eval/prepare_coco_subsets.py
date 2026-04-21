from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

COCO_ROOT = Path("eval_data/coco_raw")
IMAGES_DIR = COCO_ROOT / "val2017"
ANN_PATH = COCO_ROOT / "annotations" / "instances_val2017.json"

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

OUT_IMAGES_CORE = Path("eval_data/images/core")
OUT_LABELS_CORE = Path("eval_data/labels/core")
OUT_IMAGES_WIDE = Path("eval_data/images/wide")
OUT_LABELS_WIDE = Path("eval_data/labels/wide")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def coco_bbox_to_yolo(bbox, img_w: int, img_h: int):
    x, y, w, h = bbox
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return xc, yc, wn, hn


def main() -> None:
    if not ANN_PATH.exists():
        raise FileNotFoundError(f"Missing annotations: {ANN_PATH}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Missing images dir: {IMAGES_DIR}")

    ensure_clean_dir(OUT_IMAGES_CORE)
    ensure_clean_dir(OUT_LABELS_CORE)
    ensure_clean_dir(OUT_IMAGES_WIDE)
    ensure_clean_dir(OUT_LABELS_WIDE)

    with open(ANN_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco["categories"]
    images = coco["images"]
    annotations = coco["annotations"]

    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    core_name_to_idx = {name: i for i, name in enumerate(sorted(CORE_CLASSES))}
    wide_name_to_idx = {name: i for i, name in enumerate(sorted(WIDE_CLASSES))}

    image_id_to_meta = {img["id"]: img for img in images}
    anns_by_image = defaultdict(list)
    for ann in annotations:
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_image[ann["image_id"]].append(ann)

    core_count = 0
    wide_count = 0

    for image_id, img_meta in image_id_to_meta.items():
        file_name = img_meta["file_name"]
        img_w = img_meta["width"]
        img_h = img_meta["height"]

        anns = anns_by_image.get(image_id, [])
        if not anns:
            continue

        core_lines = []
        wide_lines = []

        for ann in anns:
            cls_name = cat_id_to_name[ann["category_id"]]

            if cls_name in CORE_CLASSES:
                xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                core_lines.append(
                    f"{core_name_to_idx[cls_name]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                )

            if cls_name in WIDE_CLASSES:
                xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                wide_lines.append(
                    f"{wide_name_to_idx[cls_name]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                )

        src_img = IMAGES_DIR / file_name

        if core_lines:
            shutil.copy2(src_img, OUT_IMAGES_CORE / file_name)
            with open(OUT_LABELS_CORE / f"{Path(file_name).stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(core_lines) + "\n")
            core_count += 1

        if wide_lines:
            shutil.copy2(src_img, OUT_IMAGES_WIDE / file_name)
            with open(OUT_LABELS_WIDE / f"{Path(file_name).stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(wide_lines) + "\n")
            wide_count += 1

    print(f"CORE images prepared: {core_count}")
    print(f"WIDE images prepared: {wide_count}")
    print("CORE class index:", core_name_to_idx)
    print("WIDE class index:", wide_name_to_idx)


if __name__ == "__main__":
    main()
