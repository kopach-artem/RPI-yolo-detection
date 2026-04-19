from __future__ import annotations

import time

import cv2


def open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def reconnect_stream(url: str, retry_delay: float = 1.0) -> cv2.VideoCapture:
    while True:
        print("WARN: frame read failed, reconnecting...")
        time.sleep(retry_delay)

        cap = open_stream(url)
        if cap.isOpened():
            print("Reconnected to stream")
            return cap


def read_frame_or_reconnect(
    cap: cv2.VideoCapture,
    url: str,
    retry_delay: float = 1.0,
):
    ok, frame = cap.read()
    if ok and frame is not None:
        return cap, frame

    cap.release()
    cap = reconnect_stream(url, retry_delay=retry_delay)
    ok, frame = cap.read()

    if not ok or frame is None:
        raise RuntimeError("Reconnected stream opened, but first frame read failed")

    return cap, frame
