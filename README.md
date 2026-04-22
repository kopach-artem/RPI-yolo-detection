# YOLO DroidCam Watch

YOLO-based live object detection for a DroidCam stream on Raspberry Pi 4 with:

* live MJPEG web stream
* periodic YOLO inference
* motion-aware alerts
* Telegram photo notifications
* saved JPEG and JSON event artifacts

## Requirements

* Debian / Raspberry Pi OS
* Git
* Internet connection
* DroidCam stream URL
* Telegram bot token and chat ID

## Project files

* `app/` — application source code
* `Dockerfile` — Docker image build file
* `start.sh` — build-if-needed and run script
* `install_docker_debian.sh` — Docker installation script
* `.env.example` — configuration template

## First setup

TODO 
data/json/
data/alerts/
```

## Notes

* `.env` is local and should not be committed
* if you change the Dockerfile or dependencies, rebuild the image
* if Docker group changes do not apply immediately, log out and log back in
