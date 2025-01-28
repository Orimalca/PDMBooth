import torch
import os, sys
from os.path import dirname


def get_arg_or_false(index):
    try:
        return sys.argv[index].lower() in ['true', '1', 't', 'y', 'yes']
    except:
        return False


# List of prompts for inference
PROMPTS = {
    "bucket": "A photo of dog in a bucket",
    "sleeping": "A photo of dog sleeping",
    "acropolis": "A photo of dog in the acropolis",
    "swimming": "A photo of dog swimming",
    "haircut": "A photo of dog getting a haircut",
    "": "A photo of dog",
}
PROMPTS_SKS = {
    "bucket": "A photo of sks dog in a bucket",
    "sleeping": "A photo of sks dog sleeping",
    "acropolis": "A photo of sks dog in the acropolis",
    "swimming": "A photo of sks dog swimming",
    "haircut": "A photo of sks dog getting a haircut",
    "": "A photo of sks dog",
}
PROMPTS_SKS = {f"sks-{key}": val for key,val in PROMPTS_SKS.items()}

# General parameters
DEVICE = 'cuda'
OBJ_NAME = sys.argv[2]
BASE_DIR = dirname(__file__)
MODEL_WEIGHTS_DIR = sys.argv[3] # e.g., "model_weights_with_prior_mask"
RESULTS_DIR_PREFIX = f"{BASE_DIR}/results/"
os.makedirs(RESULTS_DIR_PREFIX, exist_ok=True)
USE_LORA = get_arg_or_false(4)

# Set seed
SEED = int(sys.argv[1])
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
GEN = torch.Generator(device=DEVICE).manual_seed(SEED)

# Model parameters
MODEL_ID = f"{BASE_DIR}/weights/{MODEL_WEIGHTS_DIR}"
DTYPE = torch.float32 # OPTIONS: (torch.float16) / (torch.bfloat16) / (torch.float32)

# Inference parameters
gs = 7.5
num_infer_steps = 50


# Create results folder tree
r_dir = RESULTS_DIR_PREFIX
for s in MODEL_WEIGHTS_DIR.split('/'):
    r_dir = f"{r_dir}{s}/"
    os.makedirs(r_dir, exist_ok=True)

if USE_LORA:
    MODEL_NAME = sys.argv[5] # "runwayml/stable-diffusion-v1-5" or "CompVis/stable-diffusion-v1-4"
    TRAINED_TEXT_ENCODER = get_arg_or_false(6)
    if TRAINED_TEXT_ENCODER:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
        pipe.load_lora_weights(MODEL_ID)
    else:
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, safety_checker=None)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(DEVICE)
        pipe.unet.load_attn_procs(MODEL_ID)
else:
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, safety_checker=None, local_files_only=True).to(DEVICE)

for j, prompts_dict in enumerate((PROMPTS_SKS, PROMPTS)):
    for i, (k,prompt) in enumerate(prompts_dict.items()):
        img = pipe(prompt, num_inference_steps=num_infer_steps, guidance_scale=gs, generator=GEN).images[0]
        img.save(f"{r_dir}{j}.{i}.{OBJ_NAME}-{k}.png")
