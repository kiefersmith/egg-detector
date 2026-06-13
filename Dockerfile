FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-client \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir opencv-python-headless numpy ultralytics

COPY detect.py ./
COPY config.example.json ./config.json
COPY models/best.pt models/best.pt

VOLUME ["/app/snapshots", "/root/.ssh"]

ENV INTERVAL=900

CMD while true; do python detect.py; sleep "$INTERVAL"; done
