FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SCENEHOST_WORKDIR=/workspace/scenehost
ENV SCENEHOST_GAUSSIAN_SPLATTING_DIR=/opt/gaussian-splatting
ENV SCENEHOST_WORKER_VERSION=2026-05-31-reconstruction-v2
ENV QT_QPA_PLATFORM=offscreen
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

WORKDIR /workspace/scenehost

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    ffmpeg \
    colmap \
    git \
    curl \
    unzip \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @playcanvas/splat-transform@2.4.0 \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install --no-cache-dir plyfile tqdm opencv-python-headless joblib

RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git /opt/gaussian-splatting \
    && python3 -m pip install --no-cache-dir --no-build-isolation /opt/gaussian-splatting/submodules/diff-gaussian-rasterization \
    && python3 -m pip install --no-cache-dir --no-build-isolation /opt/gaussian-splatting/submodules/simple-knn

COPY handler.py .
COPY pipeline ./pipeline
COPY test_input.json .

CMD ["python3", "-u", "handler.py"]
