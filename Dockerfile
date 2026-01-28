FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git wget curl ca-certificates \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Requirements (tu peux les mettre dans requirements.txt si tu préfères)
RUN pip install --upgrade pip

RUN pip install \
    torch --index-url https://download.pytorch.org/whl/cu121 \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    diffusers[torch] \
    transformers \
    accelerate \
    safetensors \
    bitsandbytes \
    einops \
    pillow \
    pandas \
    pyarrow \
    tqdm \
    marimo \
    typer \
    oxenai \
    huggingface_hub

# Copie ton script (renomme-le si besoin)
COPY train.py /workspace/train.py

# Port facultatif si tu veux exposer marimo (UI web)
EXPOSE 8080

# Run le script en mode CLI
# Pour utiliser marimo en mode app web, change la CMD en conséquence.
CMD ["python", "train.py"]
