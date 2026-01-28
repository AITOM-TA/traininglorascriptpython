FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Dépendances Python pour le training FLUX LoRA
RUN pip install --upgrade pip && \
    pip install \
      torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install \
      torchvision \
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
      huggingface_hub \
      oxenai \
      runpod \
      requests \
      peft

# Copier le handler.py dans le conteneur
COPY handler.py /workspace/handler.py

# Point d'entrée - exécuter handler.py qui démarre le serveur RunPod
CMD ["python3", "handler.py"]
