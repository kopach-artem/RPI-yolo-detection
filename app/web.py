from __future__ import annotations

import threading
import time

from flask import Flask, Response


class MjpegWebServer:
    def __init__(self, http_port: int):
        self.http_port = http_port
        self.app = Flask(__name__)

        self._latest_frame_jpeg: bytes | None = None
        self._latest_lock = threading.Lock()

        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.route("/")
        def index():
            return """
            <html>
              <head>
                <title>YOLO DroidCam Live</title>
                <meta name="viewport" content="width=device-width, initial-scale=1" />
              </head>
              <body style="margin:0;background:#111;color:#eee;font-family:Arial,sans-serif;">
                <div style="padding:12px;">
                  <h2 style="margin:0 0 10px 0;">YOLO DroidCam Live</h2>
                  <img src="/video_feed" style="max-width:100%;height:auto;border:1px solid #444;" />
                </div>
              </body>
            </html>
            """

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self.generate_mjpeg(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def update_frame(self, jpeg_bytes: bytes | None) -> None:
        if jpeg_bytes is None:
            return
        with self._latest_lock:
            self._latest_frame_jpeg = jpeg_bytes

    def get_latest_frame(self) -> bytes | None:
        with self._latest_lock:
            return self._latest_frame_jpeg

    def generate_mjpeg(self):
        while True:
            frame = self.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    def run(self) -> None:
        self.app.run(
            host="0.0.0.0",
            port=self.http_port,
            debug=False,
            threaded=True,
        )

    def run_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread
