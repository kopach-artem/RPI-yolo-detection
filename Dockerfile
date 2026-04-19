FROM python:3.11-slim-bookworm

WORKDIR /work

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install torch==2.3.1 torchvision==0.18.1
RUN python -m pip install ultralytics flask requests

CMD ["bash"]
