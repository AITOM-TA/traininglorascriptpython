"""
Handler RunPod serverless pour l'entraînement LoRA FLUX
Point d'entrée principal pour RunPod - ne nécessite pas Marimo
"""
import os
import sys
import json
import time
import gc
import zipfile
import tempfile
import random
import math
from pathlib import Path
from datetime import datetime

# Imports PyTorch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Imports data
import numpy as np
import pandas as pd
from PIL import Image

# Imports Hugging Face
from huggingface_hub import login as hf_login

# Imports Diffusers
from diffusers import (
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    FluxPipeline
)

# Imports Transformers
from transformers import T5TokenizerFast, T5EncoderModel, CLIPTextModel, CLIPTokenizer

# Imports PEFT pour LoRA
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# Imports autres
from einops import rearrange
import bitsandbytes as bnb
from safetensors.torch import save_file
import requests

# Import RunPod
try:
    import runpod
except ImportError:
    print("❌ Erreur: runpod package non installé")
    print("Installez-le avec: pip install runpod")
    sys.exit(1)

# Import Oxen (optionnel, seulement si repo_name fourni)
try:
    from oxen import RemoteRepo
except ImportError:
    RemoteRepo = None
    print("⚠️ oxen package non installé - mode Oxen.ai désactivé")


def flush_memory():
    """Flush GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def download_and_prepare_firebase_dataset(firebase_zip_url, trigger_phrase):
    """
    Télécharge le ZIP depuis Firebase, le décompresse et crée le parquet
    
    Returns:
        tuple: (chemin_parquet, dossier_images)
    """
    # Créer un dossier temporaire
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "dataset.zip")
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Télécharger le ZIP
    print(f"📥 Téléchargement du ZIP depuis Firebase...")
    response = requests.get(firebase_zip_url, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✅ ZIP téléchargé ({os.path.getsize(zip_path) / 1024 / 1024:.2f} MB)")
    
    # Décompresser le ZIP
    print(f"📦 Décompression du ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Trouver le dossier contenant les images
    extracted_folders = [d for d in os.listdir(extract_dir) 
                         if os.path.isdir(os.path.join(extract_dir, d))]
    
    if extracted_folders:
        images_dir = os.path.join(extract_dir, extracted_folders[0])
    else:
        images_dir = extract_dir
    
    # Lister toutes les images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(Path(images_dir).glob(ext))
    
    image_files = sorted([str(f.name) for f in image_files])
    print(f"✅ Trouvé {len(image_files)} images")
    
    # Créer le fichier parquet
    captions = []
    for img_file in image_files:
        base_name = Path(img_file).stem
        caption = f"{trigger_phrase} {base_name}"
        captions.append(caption)
    
    df = pd.DataFrame({
        'image': image_files,
        'action': captions
    })
    
    parquet_path = os.path.join(temp_dir, "train.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Parquet créé avec {len(df)} entrées")
    
    return parquet_path, images_dir


class FluxDataset(Dataset):
    """Dataset for loading images and captions for Flux training"""

    def __init__(self, dataset_repo, dataset_path, images_path, resolutions=[512, 768, 1024], trigger_phrase=None):
        self.resolutions = resolutions
        self.trigger_phrase = trigger_phrase
        self.images_path = images_path

        # Si dataset_repo est vide ou None, on utilise les fichiers locaux directement
        if dataset_repo and RemoteRepo:
            self.repo = RemoteRepo(dataset_repo)
            if not os.path.exists(images_path):
                print("Downloading images")
                self.repo.download(images_path)
            if not os.path.exists(dataset_path):
                print("Downloading dataset")
                self.repo.download(dataset_path)
        else:
            # Mode Firebase ou fichiers locaux - vérifier que les fichiers existent
            if not os.path.exists(images_path):
                raise FileNotFoundError(f"Images directory not found: {images_path}")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load the dataset
        df = pd.read_parquet(dataset_path)

        # Read all images and captions
        self.image_files = []
        self.captions = []
        for index, row in df.iterrows():
            self.image_files.append(row['image'])
            self.captions.append(row['action'])

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"Found {len(self.image_files)} images in {dataset_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        caption = self.captions[idx]
        image = Image.open(os.path.join(self.images_path, image_path)).convert('RGB')

        # Add trigger word if specified and not already present
        if self.trigger_phrase and self.trigger_phrase not in caption:
            caption = f"{self.trigger_phrase}{caption}" if caption else self.trigger_phrase

        # Random resolution for multi-aspect training
        target_res = random.choice(self.resolutions)

        # Resize image maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = target_res
            new_height = int(height * target_res / width)
        else:
            new_height = target_res
            new_width = int(width * target_res / height)

        # Make dimensions divisible by 16 (Flux requirement)
        new_width = (new_width // 16) * 16
        new_height = (new_height // 16) * 16

        image = image.resize((new_width, new_height), Image.LANCZOS)
        image = self.transform(image)

        return {
            'image': image,
            'caption': caption,
            'width': new_width,
            'height': new_height
        }


def load_models(model_name, lora_rank=16, lora_alpha=16, dtype=torch.bfloat16, device="cuda"):
    """Load all FLUX models"""
    device = torch.device(device)

    # Load transformer
    print("Loading FluxTransformer2DModel")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=dtype
    )
    transformer.enable_gradient_checkpointing()

    # Apply LoRA
    print("Applying LoRA FluxTransformer2DModel")
    target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
        "proj_out"
    ]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Load VAE
    print("Loading AutoencoderKL")
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
        torch_dtype=dtype
    )
    vae.eval()

    # Load text encoders
    print("Loading CLIPTextModel")
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
        torch_dtype=dtype
    )
    text_encoder.eval()

    print("Loading T5EncoderModel")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_name,
        subfolder="text_encoder_2",
        torch_dtype=dtype
    )
    text_encoder_2.eval()

    # Load tokenizers
    print("Loading CLIPTokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer"
    )

    print("Loading T5TokenizerFast")
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        model_name,
        subfolder="tokenizer_2"
    )

    # Move models to GPU
    print("Moving models to GPU")
    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)

    return (transformer, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2)


def save_lora_weights(model, output_path, dtype=torch.float16):
    """Sauvegarde les poids LoRA en format SafeTensors"""
    lora_state_dict = get_peft_model_state_dict(model)
    lora_state_dict = {k: v.to(dtype) for k, v in lora_state_dict.items()}
    save_file(lora_state_dict, output_path)
    print(f"✅ Poids LoRA sauvegardés: {output_path}")


def write_and_save_results(transformer, repo, output_dir, lora_name, training_logs, image_logs):
    """Save training results"""
    try:
        final_save_path = os.path.join(output_dir, lora_name)
        save_lora_weights(transformer, final_save_path, torch.float16)

        # Save training logs
        training_logs_file = os.path.join(output_dir, "training_logs.jsonl")
        with open(training_logs_file, "w") as f:
            for log in training_logs:
                f.write(json.dumps(log) + "\n")

        # Save image logs
        image_logs_file = os.path.join(output_dir, "image_logs.jsonl")
        with open(image_logs_file, "w") as f:
            for log in image_logs:
                f.write(json.dumps(log) + "\n")

        if repo:
            repo.add(final_save_path, dst=output_dir)
            repo.add(training_logs_file, dst=output_dir)
            repo.add(image_logs_file, dst=output_dir)
            readme_file = os.path.join(output_dir, "README.md")
            if os.path.exists(readme_file):
                repo.add(readme_file, dst=output_dir)
            samples_dir = os.path.join(output_dir, "samples")
            if os.path.exists(samples_dir):
                repo.add(samples_dir, dst=samples_dir)
            repo.commit(f"Saving final model {output_dir}")
            print("✅ Uploaded checkpoint")
        else:
            print("⚠️ Oxen.ai non disponible - fichiers sauvegardés localement")
    except Exception as e:
        print(f"😢 Could not save weights {e}")


@torch.no_grad()
def generate_samples(config, transformer, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                    scheduler, prompts, output_dir, step, device, dtype):
    """Generate sample images during training"""
    transformer.eval()

    sample_dtype = torch.float32 if dtype == torch.bfloat16 else dtype

    vae = vae.to(device=device, dtype=sample_dtype)
    text_encoder = text_encoder.to(device=device, dtype=sample_dtype)
    text_encoder_2 = text_encoder_2.to(device=device, dtype=sample_dtype)

    pipeline = FluxPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler
    )

    pipeline = pipeline.to(device=device, dtype=sample_dtype)

    image_paths = []
    try:
        for i, prompt in enumerate(prompts):
            if "[trigger]" in prompt and config["trigger_phrase"]:
                prompt = prompt.replace("[trigger]", config["trigger_phrase"])
            elif config["trigger_phrase"] and config["trigger_phrase"] not in prompt:
                prompt = f"{config['trigger_phrase']}{prompt}"

            image = pipeline(
                prompt=prompt,
                width=config["sample_width"],
                height=config["sample_height"],
                num_inference_steps=config["sample_steps"],
                guidance_scale=config["guidance_scale"],
                generator=torch.Generator(device=device).manual_seed(42 + i),
            ).images[0]

            sample_dir = os.path.join(output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            prompt_prefix = "_".join(prompt.split(" ")[-8:])
            sample_path = os.path.join(sample_dir, f"step_{step}_sample_{i}_{prompt_prefix}.png")
            image.save(sample_path)

            print(f"Saved sample: {sample_path}")
            image_paths.append(sample_path)

    except Exception as e:
        print(f"Error generating samples: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

    finally:
        transformer.train()
        vae = vae.to(device=device, dtype=dtype)
        text_encoder = text_encoder.to(device=device, dtype=dtype)
        text_encoder_2 = text_encoder_2.to(device=device, dtype=dtype)
        flush_memory()

    return image_paths


def train(model_name, repo_name, dataset_file, images_directory, hf_api_key, trigger_phrase=None, steps=None, batch_size=None, learning_rate=None, lora_rank=None, lora_alpha=None, save_every=None, sample_every=None):
    """Main training function"""
    # Must login to Hugging Face
    hf_login(hf_api_key)

    # Default values
    dtype = torch.bfloat16
    device = torch.device("cuda")
    lora_rank = lora_rank or 16
    lora_alpha = lora_alpha or 16
    steps = steps or 2000
    batch_size = batch_size or 1
    learning_rate = learning_rate or 1e-4
    save_every = save_every or 200
    sample_every = sample_every or 200
    trigger_phrase = trigger_phrase or "Finn the dog"

    models = load_models(model_name, lora_rank, lora_alpha)

    config = {
        "trigger_phrase": trigger_phrase,
        "dataset_repo": repo_name,
        "dataset_path": dataset_file,
        "images_path": images_directory,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "steps": steps,
        "learning_rate": learning_rate,
        "optimizer": "adamw8bit",
        "noise_scheduler": "flowmatch",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "save_every": save_every,
        "sample_every": sample_every,
        "max_step_saves": 4,
        "save_dtype": "float16",
        "sample_width": 1024,
        "sample_height": 1024,
        "guidance_scale": 3.5,
        "sample_steps": 30,
        "sample_prompts": [
            "[trigger] playing chess",
            "[trigger] holding a coffee cup",
            "[trigger] DJing at a night club",
            "[trigger] wearing a blue beanie",
            "[trigger] flying a kite",
            "[trigger] fixing an upside down bicycle",
        ]
    }

    dataset = FluxDataset(
        config["dataset_repo"],
        config["dataset_path"],
        config["images_path"],
        trigger_phrase=config["trigger_phrase"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    (transformer, vae, clip_encoder, t5_encoder, clip_tokenizer, t5_tokenizer) = models
    transformer.train()

    print("Loading Noise Scheduler")
    flux_scheduler_config = {
        "shift": 3.0,
        "use_dynamic_shifting": True,
        "base_shift": 0.5,
        "max_shift": 1.15
    }
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name,
        subfolder="scheduler",
        torch_dtype=dtype
    )

    for key, value in flux_scheduler_config.items():
        if hasattr(noise_scheduler.config, key):
            setattr(noise_scheduler.config, key, value)

    print(f"Scheduler config updated: shift={noise_scheduler.config.shift}, use_dynamic_shifting={noise_scheduler.config.use_dynamic_shifting}")

    print("Setting up optimizer")
    optimizer = bnb.optim.AdamW8bit(
        transformer.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # Setup output directory (simplifié pour RunPod)
    output_dir = os.path.join("/tmp", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Setup repo (optionnel)
    repo = None
    if repo_name and RemoteRepo:
        repo = RemoteRepo(repo_name)
        experiment_prefix = f"fine-tune"
        branches = repo.branches()
        experiment_number = 0
        for branch in branches:
            if branch.name.startswith(experiment_prefix):
                experiment_number += 1
        branch_name = f"{experiment_prefix}-{config['lora_rank']}-{config['lora_alpha']}-{experiment_number}"
        print(f"Experiment name: {branch_name}")
        repo.create_checkout_branch(branch_name)
        output_dir = os.path.join("output", branch_name)
        os.makedirs(output_dir, exist_ok=True)

    config_file = os.path.join(output_dir, 'training_config.json')
    with open(config_file, 'w') as f:
        f.write(json.dumps(config))

    global_step = 0
    training_logs = []
    image_logs = []
    readme_lines = []

    readme_lines.append(f"# Flux Fine-Tune\n\n")
    readme_lines.append(f"Automatically generated during training `{model_name}`.\n\nBelow are some samples from the training run\n\n")

    # Training Loop
    while global_step < config["steps"]:
        for batch in dataloader:
            if global_step >= config["steps"]:
                break

            with torch.amp.autocast('cuda', dtype=dtype):
                images = batch['image'].to(device, dtype=dtype)
                latents = vae.encode(images).latent_dist.sample()

                scaling_factor = vae.config.scaling_factor
                shift_factor = vae.config.shift_factor
                latents = (latents - shift_factor) * scaling_factor

                clip_inputs = clip_tokenizer(
                    batch['caption'],
                    max_length=clip_tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                clip_outputs = clip_encoder(
                    input_ids=clip_inputs.input_ids.to(device),
                    attention_mask=clip_inputs.attention_mask.to(device),
                )
                pooled_embeds = clip_outputs.pooler_output

                t5_inputs = t5_tokenizer(
                    batch['caption'],
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                t5_outputs = t5_encoder(
                    input_ids=t5_inputs.input_ids.to(device),
                    attention_mask=t5_inputs.attention_mask.to(device),
                )
                prompt_embeds = t5_outputs.last_hidden_state

                noise = torch.randn_like(latents)
                num_timesteps = 1000
                t = torch.sigmoid(torch.randn((num_timesteps,), device=device))
                timesteps = ((1 - t) * num_timesteps)
                timesteps, _ = torch.sort(timesteps, descending=True)
                timesteps = timesteps.to(device=device)

                min_noise_steps = 0
                max_noise_steps = num_timesteps
                timestep_indices = torch.randint(
                    min_noise_steps,
                    max_noise_steps - 1,
                    (config['batch_size'],),
                    device=device
                ).long()

                timesteps = timesteps[timestep_indices]
                t_01 = (timesteps / num_timesteps).to(latents.device)
                noisy_latents = (1.0 - t_01) * latents + t_01 * noise

                noisy_latents = rearrange(
                    noisy_latents,
                    "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=2, pw=2
                )

                def generate_position_ids_flux(batch_size, latent_height, latent_width, device):
                    packed_h, packed_w = latent_height // 2, latent_width // 2
                    img_ids = torch.zeros(packed_h, packed_w, 3, device=device)
                    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_h, device=device)[:, None]
                    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_w, device=device)[None, :]
                    img_ids = rearrange(img_ids, "h w c -> (h w) c")
                    return img_ids

                img_ids = generate_position_ids_flux(config['batch_size'], latents.shape[2], latents.shape[3], device)
                txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device)

                guidance_embedding_scale = 1.0
                guidance = torch.tensor([guidance_embedding_scale], device=device, dtype=dtype)
                guidance = guidance.expand(latents.shape[0])

                timestep_scaled = timesteps.float() / num_timesteps

                noise_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timestep_scaled,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    guidance=guidance,
                    return_dict=False
                )[0]

                height, width = latents.shape[2], latents.shape[3]
                noise_pred = rearrange(
                    noise_pred,
                    "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=height // 2,
                    w=width // 2,
                    ph=2, pw=2,
                    c=vae.config.latent_channels
                )

                target = (noise - latents).detach()
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                t_log = {
                    "step": global_step,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                training_logs.append(t_log)
                print(t_log)

                if global_step % config["sample_every"] == 0:
                    image_paths = generate_samples(
                        config, transformer, vae, clip_encoder, t5_encoder,
                        clip_tokenizer, t5_tokenizer, noise_scheduler,
                        config["sample_prompts"],
                        output_dir, global_step, device, dtype
                    )
                    readme_lines.append(f"## Sample Images {global_step}\n\n")
                    for image_path in image_paths:
                        image_logs.append({
                            "step": global_step,
                            "image": image_path,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        if repo:
                            url = f"https://hub.oxen.ai/api/repos/{repo_name}/file/{branch_name}/{image_path}"
                            readme_lines.append(f'<a href="{url}"><img src="{url}" width="256" height="256" /></a>\n')
                        else:
                            readme_lines.append(f'<img src="{image_path}" width="256" height="256" />\n')

                    with open(os.path.join(output_dir, "README.md"), "w") as f:
                        for line in readme_lines:
                            f.write(line + "\n")

                if global_step % config["save_every"] == 0:
                    save_path = f"flux_lora_step_{global_step}.safetensors"
                    write_and_save_results(transformer, repo, output_dir, save_path, training_logs, image_logs)

                global_step += 1
                flush_memory()

    # Final save
    final_save_path = os.path.join(output_dir, "flux_lora_final.safetensors")
    write_and_save_results(transformer, repo, output_dir, "flux_lora_final.safetensors", training_logs, image_logs)

    print("Training completed!")


def handler(event):
    """Handler principal pour RunPod serverless"""
    try:
        input_data = event.get("input", {})
        
        print("🚀 Démarrage de l'entraînement LoRA FLUX...")
        print(f"Input reçu: {json.dumps(input_data, indent=2)}")
        
        # Si on utilise Firebase Storage
        if input_data.get("use_firebase") or input_data.get("firebase_zip_url"):
            firebase_zip_url = input_data.get("firebase_zip_url")
            trigger_phrase = input_data.get("trigger_phrase", "Finn the dog")
            
            if not firebase_zip_url:
                return {
                    "status": "failed",
                    "error": "firebase_zip_url is required when use_firebase is True"
                }
            
            print("🔥 Utilisation de Firebase Storage...")
            dataset_path, images_dir = download_and_prepare_firebase_dataset(
                firebase_zip_url,
                trigger_phrase
            )
            
            train(
                model_name=input_data.get("model_name", "black-forest-labs/FLUX.1-dev"),
                repo_name="",  # Pas besoin de repo Oxen.ai
                dataset_file=dataset_path,
                images_directory=images_dir,
                hf_api_key=input_data.get("hf_api_key", ""),
                trigger_phrase=trigger_phrase,
                steps=input_data.get("steps", 2000),
                batch_size=input_data.get("batch_size", 1),
                learning_rate=input_data.get("learning_rate", 1e-4),
                lora_rank=input_data.get("lora_rank", 16),
                lora_alpha=input_data.get("lora_alpha", 16),
                save_every=input_data.get("save_every", 200),
                sample_every=input_data.get("sample_every", 200)
            )
        else:
            # Mode normal avec Oxen.ai
            train(
                model_name=input_data.get("model_name", "black-forest-labs/FLUX.1-dev"),
                repo_name=input_data.get("repo_name", "ox/Finn"),
                dataset_file=input_data.get("dataset_file", "train.parquet"),
                images_directory=input_data.get("images_directory", "images"),
                hf_api_key=input_data.get("hf_api_key", ""),
                trigger_phrase=input_data.get("trigger_phrase", "Finn the dog"),
                steps=input_data.get("steps", 2000),
                batch_size=input_data.get("batch_size", 1),
                learning_rate=input_data.get("learning_rate", 1e-4),
                lora_rank=input_data.get("lora_rank", 16),
                lora_alpha=input_data.get("lora_alpha", 16),
                save_every=input_data.get("save_every", 200),
                sample_every=input_data.get("sample_every", 200)
            )
        
        return {
            "status": "completed",
            "message": "Training finished successfully"
        }
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"❌ Erreur: {error_msg}")
        print(f"Traceback: {traceback_str}")
        return {
            "status": "failed",
            "error": error_msg,
            "traceback": traceback_str
        }


# Démarrer le serveur RunPod
if __name__ == "__main__":
    print("🔧 Démarrage du handler RunPod serverless...")
    print("✅ Handler prêt à recevoir des requêtes")
    runpod.serverless.start({"handler": handler})

