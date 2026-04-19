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

Clone the repository:

```bash
git clone <REPO_URL>
cd <REPO_FOLDER>
```

Make scripts executable:

```bash
chmod +x install_docker_debian.sh start.sh
```

Install Docker:

```bash
./install_docker_debian.sh
newgrp docker
docker run hello-world
```

## Configuration

Create your local config file:

```bash
cp .env.example .env
nano .env
```

Set at least these values:

* `STREAM_URL`
* `TELEGRAM_BOT_TOKEN`
* `TELEGRAM_CHAT_ID`

## Run

Start the project:

```bash
./start.sh
```

Behavior:

* if the Docker image does not exist, it is built automatically
* if the Docker image already exists, the app starts immediately

## Open the stream

Open in your browser:

```text
http://<RASPBERRY_PI_IP>:8080
```

## Runtime data

The application writes runtime artifacts to:

```text
data/shots/
data/json/
data/alerts/
```

## Notes

* `.env` is local and should not be committed
* if you change the Dockerfile or dependencies, rebuild the image
* if Docker group changes do not apply immediately, log out and log back in
