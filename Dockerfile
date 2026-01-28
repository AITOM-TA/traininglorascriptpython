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
      oxenai

# (optionnel) tu pré-téléchargeras le modèle plus tard via un script dans le container,
# mais ici on enlève totalement le heredoc qui pose problème.

CMD ["bash"]
