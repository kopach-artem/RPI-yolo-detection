from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

METRICS_ROOT = Path("eval_runs/metrics")
PLOTS_ROOT = Path("eval_runs/plots")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["core", "wide"],
        required=True,
        help="Which subset to plot",
    )
    return parser.parse_args()


def load_summary(subset: str):
    path = METRICS_ROOT / subset / "summary_all_modes.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mode_label(item: dict) -> str:
    mode = item["run_mode"]
    model_name = Path(item["model_name"]).name
    return f"mode {mode}\n{model_name}"


def save_summary_bar_chart(metrics_all: list[dict], out_dir: Path) -> None:
    labels = [mode_label(m) for m in metrics_all]

    micro_p = [m["summary"]["micro_precision"] for m in metrics_all]
    micro_r = [m["summary"]["micro_recall"] for m in metrics_all]
    micro_f1 = [m["summary"]["micro_f1"] for m in metrics_all]
    map50 = [m["summary"]["map50"] for m in metrics_all]

    x = list(range(len(labels)))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar([i - 1.5 * width for i in x], micro_p, width=width, label="micro_precision")
    plt.bar([i - 0.5 * width for i in x], micro_r, width=width, label="micro_recall")
    plt.bar([i + 0.5 * width for i in x], micro_f1, width=width, label="micro_f1")
    plt.bar([i + 1.5 * width for i in x], map50, width=width, label="mAP50")

    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Summary metrics by mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_metrics.png", dpi=180)
    plt.close()


def save_per_class_chart(metrics_all: list[dict], out_dir: Path, metric_key: str, title: str, filename: str) -> None:
    # union of classes that appear
    class_names = []
    seen = set()
    for item in metrics_all:
        for row in item["per_class"]:
            cname = row["class_name"]
            if cname not in seen:
                seen.add(cname)
                class_names.append(cname)

    labels = [mode_label(m) for m in metrics_all]
    x = list(range(len(class_names)))
    width = 0.8 / max(1, len(metrics_all))

    plt.figure(figsize=(14, 6))

    for idx, item in enumerate(metrics_all):
        values_by_class = {row["class_name"]: row[metric_key] for row in item["per_class"]}
        values = [values_by_class.get(c, 0.0) for c in class_names]
        offset = (-0.4 + width / 2) + idx * width
        plt.bar([i + offset for i in x], values, width=width, label=labels[idx])

    plt.xticks(x, class_names, rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel(metric_key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def save_text_summary(metrics_all: list[dict], out_dir: Path) -> None:
    lines = []
    for item in metrics_all:
        s = item["summary"]
        lines.append(f"mode={item['run_mode']} model={item['model_name']}")
        lines.append(
            f"  micro_precision={s['micro_precision']:.4f} "
            f"micro_recall={s['micro_recall']:.4f} "
            f"micro_f1={s['micro_f1']:.4f} "
            f"map50={s['map50']:.4f}"
        )
        lines.append(
            f"  TP={s['total_tp']} FP={s['total_fp']} FN={s['total_fn']}"
        )
        lines.append("")

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    subset = args.subset

    metrics_all = load_summary(subset)

    out_dir = PLOTS_ROOT / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    save_summary_bar_chart(metrics_all, out_dir)
    save_per_class_chart(
        metrics_all,
        out_dir,
        metric_key="f1",
        title="Per-class F1 by mode",
        filename="per_class_f1.png",
    )
    save_per_class_chart(
        metrics_all,
        out_dir,
        metric_key="ap50",
        title="Per-class AP50 by mode",
        filename="per_class_ap50.png",
    )
    save_text_summary(metrics_all, out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
