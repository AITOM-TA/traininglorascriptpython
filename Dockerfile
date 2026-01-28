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

# --- Python deps pour ton script de training FLUX LoRA ---
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

# --- Pré-télécharger les modèles FLUX.1-dev dans le cache HF ---
# On télécharge ici pour que l'image soit déjà "warm"
RUN python - << "EOF"
from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast, T5EncoderModel

model_name = "black-forest-labs/FLUX.1-dev"

# Télécharge les sous-modèles nécessaires (ça va dans HF_HOME)
FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer")
AutoencoderKL.from_pretrained(model_name, subfolder="vae")
CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2")
CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2")

# Optionnel : pipeline complet pour être sûr que tout est OK
FluxPipeline.from_pretrained(model_name)
EOF

CMD ["bash"]
