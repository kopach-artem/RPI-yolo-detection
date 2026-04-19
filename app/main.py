from __future__ import annotations

import time

from app.alerts import (
    build_telegram_caption,
    can_send_alert,
    send_telegram_photo,
)
from app.annotate import (
    annotate_frame,
    build_counts,
    draw_status_overlay,
    encode_jpeg,
    filter_alert_detections,
    filter_moving_detections,
)
from app.config import load_config
from app.detector import YoloDetector
from app.motion import (
    create_motion_subtractor,
    detect_motion,
    enrich_detections_with_motion,
)
from app.storage import (
    make_timestamp,
    save_alert_snapshot,
    save_event_artifacts,
)
from app.stream import open_stream, read_frame_or_reconnect
from app.types import Detection, DetectionCache, FrameState
from app.web import MjpegWebServer


def clone_detections(detections: list[Detection]) -> list[Detection]:
    return [
        Detection(
            label=det.label,
            confidence=det.confidence,
            box_xyxy=list(det.box_xyxy),
            is_moving=det.is_moving,
            motion_pixels=det.motion_pixels,
            motion_ratio=det.motion_ratio,
        )
        for det in detections
    ]


def choose_stride(motion_found: bool, idle_stride: int, active_stride: int) -> int:
    return active_stride if motion_found else idle_stride


def should_run_inference(
    frame_idx: int,
    cache: DetectionCache,
    now: float,
    current_stride: int,
    detection_stale_sec: float,
) -> bool:
    return (
        frame_idx == 1
        or not cache.detections
        or (frame_idx % current_stride == 0)
        or ((now - cache.updated_at) > detection_stale_sec)
    )


def get_live_detections(
    cache: DetectionCache,
    now: float,
    detection_stale_sec: float,
) -> list[Detection]:
    if (now - cache.updated_at) <= detection_stale_sec:
        return clone_detections(cache.detections)
    return []


def mark_all_alerts_as_moving(
    detections: list[Detection],
    alert_labels: set[str],
) -> set[int]:
    moving_indexes: set[int] = set()

    for idx, det in enumerate(detections):
        if det.label in alert_labels:
            det.is_moving = True
            moving_indexes.add(idx)

    return moving_indexes


def update_fps(state: FrameState, now: float) -> float:
    if state.last_time <= 0:
        state.last_time = now
        return state.fps

    dt = now - state.last_time
    if dt > 0:
        current_fps = 1.0 / dt
        state.fps = current_fps if state.fps == 0 else (0.9 * state.fps + 0.1 * current_fps)

    state.last_time = now
    return state.fps


def print_startup_info(cfg) -> None:
    print("Stream:", cfg.stream_url)
    print("Model:", cfg.model_name)
    print("Alert labels:", sorted(cfg.alert_labels))
    print("Web UI: http://<RASPBERRY_PI_IP>:%d" % cfg.http_port)
    print("Save every:", cfg.save_every, "sec")
    print("Telegram cooldown:", cfg.telegram_cooldown, "sec")
    print("Idle frame stride:", cfg.idle_frame_stride)
    print("Active frame stride:", cfg.active_frame_stride)
    print("Motion gate:", cfg.motion_gate)
    print("Motion min area:", cfg.motion_min_area)
    print("Motion box pad:", cfg.motion_box_pad)
    print("Motion box min pixels:", cfg.motion_box_min_pixels)
    print("Motion box min ratio:", cfg.motion_box_min_ratio)
    print("Press Ctrl+C to stop")


def main() -> None:
    cfg = load_config()

    detector = YoloDetector(cfg)
    motion_subtractor = create_motion_subtractor()
    web = MjpegWebServer(cfg.http_port)

    cap = open_stream(cfg.stream_url)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open stream: {cfg.stream_url}")

    web.run_in_background()

    state = FrameState(last_time=time.time())
    cache = DetectionCache()

    print_startup_info(cfg)

    try:
        while True:
            cap, frame = read_frame_or_reconnect(cap, cfg.stream_url)

            state.frame_idx += 1
            now = time.time()
            frame_h, frame_w = frame.shape[:2]

            motion_found = True
            motion_area = 0
            motion_mask = None

            if cfg.motion_gate:
                motion_result = detect_motion(
                    frame,
                    motion_subtractor,
                    min_area=cfg.motion_min_area,
                )
                motion_found = motion_result.found
                motion_area = motion_result.area
                motion_mask = motion_result.mask

                if state.warmup_count < cfg.motion_warmup_frames:
                    state.warmup_count += 1
                    motion_found = False
            else:
                motion_found = True

            current_stride = choose_stride(
                motion_found=motion_found,
                idle_stride=cfg.idle_frame_stride,
                active_stride=cfg.active_frame_stride,
            )

            if should_run_inference(
                frame_idx=state.frame_idx,
                cache=cache,
                now=now,
                current_stride=current_stride,
                detection_stale_sec=cfg.detection_stale_sec,
            ):
                cache.detections = detector.predict(frame)
                cache.updated_at = now
                state.last_infer_time = now

            detections = get_live_detections(
                cache=cache,
                now=now,
                detection_stale_sec=cfg.detection_stale_sec,
            )

            if cfg.motion_gate:
                detections, moving_indexes = enrich_detections_with_motion(
                    detections=detections,
                    motion_mask=motion_mask,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    pad=cfg.motion_box_pad,
                    min_pixels=cfg.motion_box_min_pixels,
                    min_ratio=cfg.motion_box_min_ratio,
                )
            else:
                moving_indexes = mark_all_alerts_as_moving(
                    detections=detections,
                    alert_labels=cfg.alert_labels,
                )

            alert_detections = filter_alert_detections(detections, cfg.alert_labels)
            moving_alert_detections = filter_moving_detections(detections, cfg.alert_labels)

            alert_counts = build_counts(alert_detections)
            moving_alert_counts = build_counts(moving_alert_detections)

            alert_seen = len(alert_detections) > 0
            moving_alert_hit = len(moving_alert_detections) > 0

            annotated = annotate_frame(
                frame=frame,
                detections=detections,
                alert_labels=cfg.alert_labels,
                moving_indexes=moving_indexes,
            )

            fps = update_fps(state, now)

            annotated = draw_status_overlay(
                frame=annotated,
                fps=fps,
                detections_count=len(detections),
                motion_found=motion_found,
                motion_area=motion_area,
                model_name=cfg.model_name,
                current_stride=current_stride,
                alert_counts=alert_counts if alert_seen else None,
                moving_alert_counts=moving_alert_counts if moving_alert_hit else None,
            )

            jpeg_bytes = encode_jpeg(annotated, cfg.jpeg_quality)
            web.update_frame(jpeg_bytes)

            if moving_alert_hit and (now - state.last_save >= cfg.save_every):
                state.last_save = now
                ts = make_timestamp()

                jpg_path, _json_path = save_event_artifacts(
                    cfg=cfg,
                    ts=ts,
                    annotated_frame=annotated,
                    detections=detections,
                    alert_detections=alert_detections,
                    moving_alert_detections=moving_alert_detections,
                    motion_found=motion_found,
                    motion_area=motion_area,
                )

                print(f"[{ts}] saved: {jpg_path.name}, moving_alerts={moving_alert_counts}")

            if moving_alert_hit and can_send_alert(
                now=now,
                last_alert_time=state.last_alert,
                cooldown=cfg.telegram_cooldown,
            ):
                ts = make_timestamp()

                alert_path = save_alert_snapshot(
                    cfg=cfg,
                    ts=ts,
                    annotated_frame=annotated,
                    moving_alert_detections=moving_alert_detections,
                )

                caption = build_telegram_caption(
                    detections=moving_alert_detections,
                    model_name=cfg.model_name,
                    ts=ts,
                )

                send_telegram_photo(cfg, alert_path, caption)
                state.last_alert = now

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
