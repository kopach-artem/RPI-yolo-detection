from __future__ import annotations

from datetime import datetime
from pathlib import Path

import requests

from app.config import AppConfig
from app.types import Detection


def can_send_alert(now: float, last_alert_time: float, cooldown: float) -> bool:
    return (now - last_alert_time) >= cooldown


def build_alert_filename(ts: str, detections: list[Detection]) -> str:
    labels = sorted({det.label for det in detections})
    labels_part = "_".join(labels) if labels else "unknown"
    return f"{ts}_{labels_part}.jpg"


def build_telegram_caption(
    detections: list[Detection],
    model_name: str,
    ts: str | None = None,
) -> str:
    if ts is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    counts = build_counts(detections)
    counts_text = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    if not counts_text:
        counts_text = "no objects"

    return (
        "Moving alert detected\n"
        f"time: {ts}\n"
        f"labels: {counts_text}\n"
        f"model: {model_name}"
    )


def build_counts(detections: list[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return counts


def send_telegram_photo(
    cfg: AppConfig,
    photo_path: Path,
    caption: str,
) -> bool:
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        print("WARN: Telegram token/chat_id not set, skip alert")
        return False

    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendPhoto"

    try:
        with open(photo_path, "rb") as f:
            resp = requests.post(
                url,
                data={
                    "chat_id": cfg.telegram_chat_id,
                    "caption": caption,
                },
                files={
                    "photo": f,
                },
                timeout=30,
            )
    except requests.RequestException as exc:
        print("Telegram send exception:", exc)
        return False
    except OSError as exc:
        print("Telegram photo open exception:", exc)
        return False

    if resp.ok:
        print("Telegram alert sent")
        return True

    print("Telegram send failed:", resp.status_code, resp.text)
    return False
