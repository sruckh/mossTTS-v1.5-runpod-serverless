# Multi-stage production build starting with Ubuntu 24.04 & CUDA 12.8.2
FROM nvidia/cuda:12.8.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HOME=/runpod-volume/huggingface \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    MODEL_REPO=OpenMOSS-Team/MOSS-TTS-v1.5 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies & Python 3.12 (native Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Setup Virtual Environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch 2.8.0 + torchvision 0.23.0 + torchaudio 2.8.0 for CUDA 12.8
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install Pre-compiled FlashAttention-2 wheel matching PyTorch 2.8 + CUDA 12 ABI
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Install MOSS-TTS, RunPod SDK & Accelerated HuggingFace Hub ('hf' CLI tool + hf_transfer)
WORKDIR /app
RUN git clone https://github.com/OpenMOSS/MOSS-TTS.git . && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e . && \
    pip install runpod soundfile scipy "huggingface_hub[cli]" hf_transfer "transformers>=4.48.0"

# Copy worker handler script
COPY handler.py /app/handler.py

# Entrypoint for RunPod Serverless Worker
CMD ["python3", "-u", "/app/handler.py"]
