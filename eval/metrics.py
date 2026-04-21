from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

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

LABELS_ROOT = Path("eval_data/labels")
PREDICTIONS_ROOT = Path("eval_runs/images")
METRICS_ROOT = Path("eval_runs/metrics")


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
        help="Comma-separated modes. Example: 0,1,2,3",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching",
    )
    return parser.parse_args()


def default_modes_for_subset(subset: str) -> list[int]:
    if subset == "core":
        return [0, 1, 2, 3]
    if subset == "wide":
        return [0, 1, 3]
    raise ValueError(f"Unsupported subset: {subset}")


def parse_modes(modes_arg: str | None, subset: str) -> list[int]:
    if not modes_arg:
        return default_modes_for_subset(subset)

    modes = []
    for part in modes_arg.split(","):
        part = part.strip()
        if not part:
            continue
        modes.append(int(part))
    return modes


def class_names_for_subset(subset: str) -> list[str]:
    if subset == "core":
        return sorted(CORE_CLASSES)
    if subset == "wide":
        return sorted(WIDE_CLASSES)
    raise ValueError(f"Unsupported subset: {subset}")


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    bw = w * img_w
    bh = h * img_h
    cx = xc * img_w
    cy = yc * img_h

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return [x1, y1, x2, y2]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_ground_truth(subset: str, class_names: list[str]) -> dict[str, dict[str, list[list[float]]]]:
    """
    Returns:
        gt[class_name][image_name] = list of gt boxes in xyxy
    """
    labels_dir = LABELS_ROOT / subset
    images_dir = Path(f"eval_data/images/{subset}")

    class_idx_to_name = {i: name for i, name in enumerate(class_names)}
    gt: dict[str, dict[str, list[list[float]]]] = {name: {} for name in class_names}

    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        image_stem = label_path.stem
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = images_dir / f"{image_stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        image_name = image_path.name

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue

            cls_idx = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            if cls_idx not in class_idx_to_name:
                continue

            cls_name = class_idx_to_name[cls_idx]
            box = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)

            gt.setdefault(cls_name, {}).setdefault(image_name, []).append(box)

    return gt


def load_predictions(subset: str, mode: int, class_names: list[str]) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """
    Returns:
        model_name, preds[class_name] = list of {"image_name", "confidence", "box_xyxy"}
    """
    pred_path = PREDICTIONS_ROOT / subset / f"mode_{mode}.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds: dict[str, list[dict[str, Any]]] = {name: [] for name in class_names}

    for item in data["predictions"]:
        image_name = item["image_name"]
        for det in item["detections"]:
            cls_name = det["label"]
            if cls_name not in preds:
                continue

            preds[cls_name].append(
                {
                    "image_name": image_name,
                    "confidence": float(det["confidence"]),
                    "box_xyxy": [float(v) for v in det["box_xyxy"]],
                }
            )

    return data["model_name"], preds


def compute_ap(tp_flags: list[int], fp_flags: list[int], gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    if not tp_flags:
        return 0.0

    cum_tp = []
    cum_fp = []
    tp_running = 0
    fp_running = 0

    for tp, fp in zip(tp_flags, fp_flags):
        tp_running += tp
        fp_running += fp
        cum_tp.append(tp_running)
        cum_fp.append(fp_running)

    recalls = []
    precisions = []
    for tp_c, fp_c in zip(cum_tp, cum_fp):
        recall = tp_c / gt_count
        precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def evaluate_class(
    class_name: str,
    gt_for_class: dict[str, list[list[float]]],
    preds_for_class: list[dict[str, Any]],
    iou_thr: float,
) -> dict[str, Any]:
    gt_count = sum(len(v) for v in gt_for_class.values())

    matched = {
        image_name: [False] * len(boxes)
        for image_name, boxes in gt_for_class.items()
    }

    preds_sorted = sorted(preds_for_class, key=lambda x: x["confidence"], reverse=True)

    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for pred in preds_sorted:
        image_name = pred["image_name"]
        pred_box = pred["box_xyxy"]

        gt_boxes = gt_for_class.get(image_name, [])
        best_iou = 0.0
        best_idx = -1

        for idx, gt_box in enumerate(gt_boxes):
            if matched[image_name][idx]:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_thr:
            matched[image_name][best_idx] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp = sum(tp_flags)
    fp = sum(fp_flags)
    fn = gt_count - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / gt_count if gt_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    ap50 = compute_ap(tp_flags, fp_flags, gt_count)

    return {
        "class_name": class_name,
        "gt_count": gt_count,
        "pred_count": len(preds_for_class),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "ap50": round(ap50, 6),
    }


def summarize_metrics(per_class_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(m["tp"] for m in per_class_metrics)
    total_fp = sum(m["fp"] for m in per_class_metrics)
    total_fn = sum(m["fn"] for m in per_class_metrics)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    valid_ap = [m["ap50"] for m in per_class_metrics if m["gt_count"] > 0]
    map50 = sum(valid_ap) / len(valid_ap) if valid_ap else 0.0

    macro_precision_vals = [m["precision"] for m in per_class_metrics if m["gt_count"] > 0]
    macro_recall_vals = [m["recall"] for m in per_class_metrics if m["gt_count"] > 0]
    macro_f1_vals = [m["f1"] for m in per_class_metrics if m["gt_count"] > 0]

    macro_precision = sum(macro_precision_vals) / len(macro_precision_vals) if macro_precision_vals else 0.0
    macro_recall = sum(macro_recall_vals) / len(macro_recall_vals) if macro_recall_vals else 0.0
    macro_f1 = sum(macro_f1_vals) / len(macro_f1_vals) if macro_f1_vals else 0.0

    return {
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "map50": round(map50, 6),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def evaluate_mode(subset: str, mode: int, iou_thr: float) -> dict[str, Any]:
    class_names = class_names_for_subset(subset)
    gt = load_ground_truth(subset, class_names)
    model_name, preds = load_predictions(subset, mode, class_names)

    per_class = []
    for class_name in class_names:
        metrics = evaluate_class(
            class_name=class_name,
            gt_for_class=gt.get(class_name, {}),
            preds_for_class=preds.get(class_name, []),
            iou_thr=iou_thr,
        )
        per_class.append(metrics)

    summary = summarize_metrics(per_class)

    return {
        "subset": subset,
        "run_mode": mode,
        "model_name": model_name,
        "iou_threshold": iou_thr,
        "summary": summary,
        "per_class": per_class,
    }


def print_report(metrics: dict[str, Any]) -> None:
    summary = metrics["summary"]
    print(f"\n=== subset={metrics['subset']} mode={metrics['run_mode']} ===")
    print("model:", metrics["model_name"])
    print(
        "micro P/R/F1:",
        summary["micro_precision"],
        summary["micro_recall"],
        summary["micro_f1"],
    )
    print(
        "macro P/R/F1:",
        summary["macro_precision"],
        summary["macro_recall"],
        summary["macro_f1"],
    )
    print("mAP50:", summary["map50"])
    print("TP/FP/FN:", summary["total_tp"], summary["total_fp"], summary["total_fn"])

    print("\nPer-class:")
    for m in metrics["per_class"]:
        print(
            f"  {m['class_name']}: "
            f"P={m['precision']:.4f} "
            f"R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} "
            f"AP50={m['ap50']:.4f} "
            f"(TP={m['tp']} FP={m['fp']} FN={m['fn']}, GT={m['gt_count']}, Pred={m['pred_count']})"
        )


def main():
    args = parse_args()
    subset = args.subset
    modes = parse_modes(args.modes, subset)

    out_dir = METRICS_ROOT / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = []

    for mode in modes:
        metrics = evaluate_mode(subset=subset, mode=mode, iou_thr=args.iou)
        print_report(metrics)

        out_path = out_dir / f"metrics_mode_{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        combined.append(metrics)
        print(f"Saved metrics to: {out_path}")

    combined_path = out_dir / "summary_all_modes.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\nSaved combined summary to: {combined_path}")


if __name__ == "__main__":
    main()
