import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DiffusionPipeline, DPMSolverMultistepScheduler,
    StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor, AttnAddedKVProcessor2_0, LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor, LoRAAttnProcessor2_0, SlicedAttnAddedKVProcessor)
from diffusers.optimization import get_scheduler
from diffusers.utils import TEXT_ENCODER_ATTN_MODULE, check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from StableDiffusionPipelineWithDDIMInversion import StableDiffusionPipelineWithDDIMInversion
from diffusers import DDIMScheduler
from attention_store import AttentionStore
import ptp_utils
from transformers import SamModel, SamProcessor
from torchvision.transforms.functional import crop as torchvision_crop

if is_wandb_available():
    import wandb
    import time
    from datetime import datetime
    from wandb.sdk.lib.runid import generate_id as wnb_generate_id
    # metrics libraries
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from transformers import ViTModel
check_min_version("0.18.0") # error if minimal version of diffusers not installed (remove at your own risks)
logger = get_logger(__name__)




def extract_pdm_fs(pipe, img, prompt, device='cuda'):
    ctrl = AttentionStore()
    num_inv_steps, num_recon_steps = 999, 999
    inv_latents, _ = pipe.invert(prompt, image=img, num_inference_steps=num_inv_steps,
                                 guidance_scale=1.0, deterministic=True)
    ptp_utils.register_attention_control_efficient_pdmbooth(pipe, ctrl)
    recon = pipe(prompt, latents=inv_latents, include_zero_timestep=True, guidance_scale=1.0,
                 num_inference_steps=num_recon_steps)
    layer_name = 'up_blocks_3_attentions_2_transformer_blocks_0_attn1_self' # layer name
    l_key = f"Q_{layer_name}"
    reversed_Q_features = [ctrl.attn_store[i][l_key][0] for i in ctrl.attn_store]
    reversed_Q_features = torch.stack(reversed_Q_features).to(device)
    return reversed_Q_features.flip(dims=[0]) # i=0 is t=num_rec_steps and i=(num_rec_steps-1) is t=0


def save_model_card(repo_id: str, imgs=None, base_model=str, train_text_encoder=False,
                    prompt=str, repo_folder=None, pipe: DiffusionPipeline = None):
    img_str = ""
    for i, img in enumerate(imgs):
        img.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"
    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
inst_prompt: {prompt}
tags:
- {'stable-diffusion' if isinstance(pipe, StableDiffusionPipeline) else 'if'}
- {'stable-diffusion-diffusers' if isinstance(pipe, StableDiffusionPipeline) else 'if-diffusers'}
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_cls_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    model_cls = text_encoder_config.architectures[0]

    if model_cls == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_cls == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    elif model_cls == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_cls} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name", type=str, default=None,
        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--inst_data_dir", type=str, default=None, required=True,
        help="A folder containing the training data of instance images.")
    parser.add_argument("--cls_data_dir", type=str, default=None, required=False,
        help="A folder containing the training data of class images.")
    parser.add_argument("--inst_prompt", type=str, default=None, required=True,
        help="The prompt with identifier specifying the instance")
    parser.add_argument("--cls_prompt", type=str, default=None,
        help="The prompt to specify images in the same class as provided instance images.")
    parser.add_argument("--val_prompt", type=str, default=None,
        help="A prompt that is used during validation to verify that the model is learning.")
    parser.add_argument("--num_val_imgs", type=int, default=4,
        help="Number of images that should be generated during validation with `val_prompt`.")
    parser.add_argument("--val_epochs", type=int, default=50,
        help="Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt `args.val_prompt` multiple times: `args.num_val_imgs`.")
    parser.add_argument("--use_prior_loss", default=False, action="store_true",
        help="Flag to add prior preservation loss.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0,
        help="The weight of prior preservation loss.")
    parser.add_argument("--num_cls_imgs", type=int, default=100,
        help="Minimal class images for prior preservation loss. If there are not enough images already present in cls_data_dir, additional images will be sampled with cls_prompt.")
    parser.add_argument("--out_dir", type=str, default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,
        help="The resolution for input images, all the images in the train/val dataset will be resized to this resolution")
    parser.add_argument("--center_crop", default=False, action="store_true",
        help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")
    parser.add_argument("--train_text_encoder", action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    parser.add_argument("--train_batch_size", type=int, default=4,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--sample_batch_size", type=int, default=4,
        help="Batch size (per device) for sampling images.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--ckpting_steps", type=int, default=500,
        help="Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_ckpt`.")
    parser.add_argument("--ckpts_total_limit", type=int, default=None,
        help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_ckpt", type=str, default=None,
        help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--ckpting_steps`, or `'latest'` to automatically select the last available checkpoint.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_ckpting", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--lr", type=float, default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0,
        help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dl_num_workers", type=int, default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--use_8bit_adam", action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_eps", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true",
        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None,
        help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None,
        help="The name of the repository to keep in sync with the local `out_dir`.",)
    parser.add_argument("--log_dir", type=str, default="logs",
        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *out_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--allow_tf32", action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    parser.add_argument(
        "--prior_generation_precision", type=str, default=None, choices=["no", "fp32", "fp16", "bf16"],
        help="Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
        help="Whether or not to use xformers.")
    parser.add_argument("--precompute_text_embeds", action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.")
    parser.add_argument("--tokenizer_max_length", type=int, default=None, required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.")
    parser.add_argument("--text_encoder_use_attn_mask", action="store_true", required=False,
        help="Whether to use attention mask for the text encoder")
    parser.add_argument("--val_imgs", required=False, default=None, nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.")
    parser.add_argument("--cls_labels_conditioning", required=False, default=None,
        help="The optional `cls_label` conditioning to pass to the unet, available values are `timesteps`.")
    parser.add_argument("--rank", type=int, default=4,
        help="The dimension of the LoRA update matrices.")
    parser.add_argument("--features_data_dir", type=str, default=None,
        help="Relative path to dir containing the PDM features of the instance images.")
    parser.add_argument("--save_pdm_fs", default=False, action="store_true",
        help="Include this argument to prevent storing PDM features when they are being generated")
    parser.add_argument("--use_inst_loss", default=False, action="store_true", help="Use instance loss.")
    parser.add_argument("--use_pdm_loss", default=False, action="store_true", help="Flag to add pdm loss.")
    parser.add_argument("--pdm_loss_weight", type=float, default=0.05, help="The weight of pdm loss.")
    parser.add_argument("--mask_pdm", default=False, action="store_true", help='Flag to mask pdm loss')
    parser.add_argument("--mask_dm", default=False, action="store_true", help='Flag to mask diffusion loss')
    parser.add_argument("--mask_prior", default=False, action="store_true", help='Flag to mask prior loss')
    parser.add_argument("--save_pdm_masks",default=False,action="store_true",help="Flag to save pdm masks.")
    parser.add_argument("--del_cls_imgs_dir", default=False, action="store_true",
        help="Whether to delete class images dir after training is done.")
    parser.add_argument("--trackers_proj_name", required=False, default='PDMBooth', help="Tracker project name.")
    parser.add_argument("--wandb_exp", required=False, default=None, help="Wandb experiment name.")
    parser.add_argument("--wandb_dir", required=False, default='', help="Wandb experiment dir name.")
    parser.add_argument("--test_prompts", action='append', default=[], help='Add prompt to test list')
    parser.add_argument("--num_test_imgs_per_prompt", type=int, default=4,
        help="Number of images that should be generated during test for each prompt in `test_prompts`.")
    args = parser.parse_args(input_args) if input_args is not None else parser.parse_args()
    if not args.test_prompts and args.val_prompt is not None:
        args.test_prompts = [args.val_prompt]

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.use_prior_loss:
        if args.cls_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.cls_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else: # logger is not available yet
        if args.cls_data_dir is not None:
            warnings.warn("You need not use --cls_data_dir without --use_prior_loss.")
        if args.cls_prompt is not None:
            warnings.warn("You need not use --cls_prompt without --use_prior_loss.")

    if args.train_text_encoder and args.precompute_text_embeds:
        raise ValueError("`--train_text_encoder` cannot be used with `--precompute_text_embeds`")

    return args


class PDMBoothDataset(Dataset):
    """A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts."""
    def __init__(
        self, inst_data_root, inst_prompt, tokenizer, cls_data_root=None, cls_prompt=None, cls_num=None,
        size=512, center_crop=False, encoder_hidden_states=None, inst_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None, inst_features_root: str = None, save_pdm_fs: bool = None,
        do_extract_pdm_fs: bool = False, sam_processor: SamProcessor = None, sam_model: SamModel = None,
        device: torch.device | str = None
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.inst_prompt_encoder_hidden_states = inst_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self.device = device
        self.sam_processor, self.sam_model = sam_processor, sam_model

        self.inst_data_root = Path(inst_data_root)
        if not self.inst_data_root.exists():
            raise ValueError(f"Instance {self.inst_data_root} images root doesn't exists.")

        self.inst_imgs, self.pdm_imgs = [[] for _ in range(2)]
        self.inst_masks, self.pdm_masks = [[] if c else None for c in [args.mask_dm, args.mask_pdm]]
        self.pdm_fs = [] if args.use_pdm_loss else None

        self.save_pdm_fs = save_pdm_fs
        self.do_extract_pdm_fs = do_extract_pdm_fs
        if not inst_features_root or not Path(inst_features_root).exists():
            self.do_extract_pdm_fs = True
            self.inst_features_root = Path(str(self.inst_data_root) + "_pdm_features")
            logger.info(
                "Directory for PDM features of instance images not found (features will be generated " \
                + (f"and saved to {self.inst_features_root}" if save_pdm_fs \
                else "but won't be saved (locally) afterwards; if you wish to save them set save_pdm_fs argument to True") \
                + ")"
            )
        else:
            self.inst_features_root = inst_features_root
            self.inst_features_path = list(Path(self.inst_features_root).iterdir())
            logger.info("Directory for PDM features of instance images found. PDM features will be loaded from there.")

        self.inst_imgs_path = list(Path(inst_data_root).iterdir())
        self.num_inst_imgs = len(self.inst_imgs_path)
        for i, img_path in enumerate(self.inst_imgs_path): # Renames instances images to indices
            img_path.rename(img_path.parent / f"{i}{img_path.suffix}") if img_path.stem != str(i) else ()
        self.inst_imgs_path = list(Path(inst_data_root).iterdir()) # update path list based on new names

        self.inst_prompt = inst_prompt
        self._length = self.num_inst_imgs

        if cls_data_root is not None:
            self.cls_data_root = Path(cls_data_root)
            self.cls_data_root.mkdir(parents=True, exist_ok=True)
            self.cls_imgs_path = list(self.cls_data_root.iterdir())
            if cls_num is not None:
                self.num_cls_imgs = min(len(self.cls_imgs_path), cls_num)
            else:
                self.num_cls_imgs = len(self.cls_imgs_path)
            self._length = max(self.num_cls_imgs, self.num_inst_imgs)
            self.cls_prompt = cls_prompt
        else:
            self.cls_data_root = None

        self.to_tensor = transforms.ToTensor()
        # Mask Transformations
        mresize_temp = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
        self.mresize = lambda mask_pil: self.fix_mask(mresize_temp(mask_pil))
        mresize64_temp = transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC)
        self.mresize64 = lambda mask_pil: self.fix_mask(mresize64_temp(mask_pil))
        # Image Transformations
        self.iresize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.iresize64 = transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR)
        self.ccrop = transforms.CenterCrop(size)
        self.rcrop = transforms.RandomCrop(size, padding=None, pad_if_needed=False) # same rand crop
        self.img_transforms = transforms.Compose([self.to_tensor, transforms.Normalize([0.5],[0.5])])

    def __len__(self):
        return self._length

    def fix_mask(self, mask_p):
        """Function to fix a mask after a resize operation (NOTE: expects values range of [0,255])"""
        mask_t = self.to_tensor(mask_p)[0] # (H,W)
        mask_t[mask_t > 0] = 1.; mask_t[mask_t < 0] = 0.
        mask_p = transforms.ToPILImage(mode='L')(mask_t)
        return mask_p

    def get_mask(self, img, save=False, dir_n=None, mask_n=None, ext="png", in_pts=None, resize=True):
        inputs = self.sam_processor(img, input_points=in_pts, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())

        best_mask_idx = torch.argmax(outputs.iou_scores[0,0]).item() # (bs,1,n_masks) -> (n_masks) -> argmax
        best_mask_idx = 1 # NOTE: using a fixed index
        mask_b = masks[0][0,best_mask_idx] # (H,W)
        mask_b = (~mask_b) # invert mask if needed
        mask_t = mask_b.float()
        mask_p = transforms.ToPILImage(mode='L')(mask_t)
        mask_p = self.mresize(mask_p) if resize else mask_p
        if save:
            os.makedirs(dir_n, exist_ok=True)
            mask_p.save(f'{dir_n}/{mask_n}.{ext}')
        return mask_p

    def __getitem__(self, index):
        ex = {} # creating example for current index example
        ex["inst_img_idx"] = index % self.num_inst_imgs

        if args.use_pdm_loss:
            ex["pdm_imgs"] = self.img_transforms(self.pdm_imgs[ex["inst_img_idx"]]).to(self.device)

        inst_img = self.inst_imgs[ex["inst_img_idx"]]
        i,j,h,w = self.rcrop.get_params(inst_img, self.rcrop.size) # save rand crop boundings
        inst_img = torchvision_crop(inst_img, i,j,h,w) # rand crop image to (3, self.size, self.size)
        ex['inst_imgs'] = self.img_transforms(inst_img).to(self.device)
        if args.mask_dm:
            inst_mask = self.inst_masks[ex["inst_img_idx"]]
            inst_mask = torchvision_crop(inst_mask, i,j,h,w) # rand crop using the same boundings
            inst_mask = self.to_tensor(self.mresize64(inst_mask)) # resize to 64x64 -> fix
            ex["inst_mask"] = inst_mask.unsqueeze(0).bool().to(self.device).repeat(1,4,1,1) # [1,4,64,64]
        if self.encoder_hidden_states is not None:
            ex["inst_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.inst_prompt, tokenizer_max_length=self.tokenizer_max_length)
            ex["inst_prompt_ids"] = text_inputs.input_ids
            ex["inst_attn_mask"] = text_inputs.attention_mask

        if self.cls_data_root:
            ex["cls_img_idx"] = index % self.num_cls_imgs
            cls_img = exif_transpose(Image.open(self.cls_imgs_path[ex["cls_img_idx"]]))
            cls_img = cls_img.convert("RGB") if not cls_img.mode == "RGB" else cls_img
            ex["cls_imgs"] = self.img_transforms(cls_img).to(self.device)
            if args.mask_prior:
                cls_mask = self.to_tensor(self.mresize64(self.get_mask(cls_img, resize=False)))
                ex["cls_mask"] = cls_mask.unsqueeze(0).bool().to(self.device).repeat(1,4,1,1) # [1,4,64,64]
            if self.inst_prompt_encoder_hidden_states is not None:
                ex["cls_prompt_ids"] = self.inst_prompt_encoder_hidden_states
            else:
                cls_text_inputs = tokenize_prompt(
                    self.tokenizer, self.cls_prompt, tokenizer_max_length=self.tokenizer_max_length)
                ex["cls_prompt_ids"] = cls_text_inputs.input_ids
                ex["cls_attn_mask"] = cls_text_inputs.attention_mask

        return ex


def collate_fn(examples):
    has_attn_mask = "inst_attn_mask" in examples[0]

    input_ids = [ex["inst_prompt_ids"] for ex in examples]
    pixel_values = [ex["inst_imgs"] for ex in examples]
    inst_imgs_indices = [ex["inst_img_idx"] for ex in examples]
    inst_imgs_masks = [ex["inst_mask"] for ex in examples] if args.mask_dm else None

    if has_attn_mask:
        attn_mask = [ex["inst_attn_mask"] for ex in examples]

    if args.use_prior_loss: # Concat examples (to avoid doing multiple forward passes).
        input_ids += [ex["cls_prompt_ids"] for ex in examples]
        pixel_values += [ex["cls_imgs"] for ex in examples]
        cls_imgs_masks = [ex["cls_mask"] for ex in examples] if args.mask_prior else None

        if has_attn_mask:
            attn_mask += [ex["cls_attn_mask"] for ex in examples]

    if args.use_pdm_loss:
        pixel_values += [ex["pdm_imgs"] for ex in examples]
        input_ids += [ex["inst_prompt_ids"] for ex in examples]
        if has_attn_mask:
            attn_mask += [ex["inst_attn_mask"] for ex in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "inst_imgs_indices": inst_imgs_indices,
        "inst_imgs_masks": inst_imgs_masks if args.use_inst_loss and args.mask_dm else None,
        "cls_imgs_masks": cls_imgs_masks if args.use_prior_loss and args.mask_prior else None,
    }
    if has_attn_mask:
        batch["attn_mask"] = torch.cat(attn_mask, dim=0)

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."
    def __init__(self, prompt, num_samples):
        self.prompt, self.num_samples = prompt, num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, index):
        ex = {}
        ex["prompt"], ex["index"] = self.prompt, index
        return ex

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    max_length = tokenizer_max_length if tokenizer_max_length is not None else tokenizer.model_max_length
    text_inputs = tokenizer(
        prompt, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return text_inputs
def encode_prompt(text_encoder, input_ids, attn_mask, text_encoder_use_attn_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)
    attn_mask = attn_mask.to(text_encoder.device) if text_encoder_use_attn_mask else None
    prompt_embeds = text_encoder(text_input_ids, attention_mask=attn_mask)
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds


def main(args):
    if any([not args.use_inst_loss and args.mask_dm,
            not args.use_pdm_loss and args.mask_pdm,
            not args.use_prior_loss and args.mask_prior]):
        raise ValueError("Don't set one of the loss arguments to False and its corresponding mask argument to True (For example, args.use_pdm_loss=False and args.mask_pdm=True). Please avoid doing it")

    if args.report_to in ('wandb', 'all'):
        if not is_wandb_available():
            raise ImportError("'wandb' is registered as a tracker but not installed")
        wnb_x_start_time, wnb_id = None, None
        if not args.resume_from_ckpt: # a new 'wandb' run --> add experiment date and id to out_dir name
            wnb_x_start_time = time.time()
            wnb_start_datetime = datetime.fromtimestamp(wnb_x_start_time).strftime("%Y%m%d_%H%M%S")
            wnb_id = wnb_generate_id()
            args.out_dir = args.out_dir[:-1] if args.out_dir.endswith('/') else args.out_dir # remove end "/"
            args.out_dir = f"{args.out_dir}--{wnb_start_datetime}-{wnb_id}"

    BASE_DIR = os.path.dirname(__file__)
    out_path = args.out_dir
    if args.out_dir.startswith('./'):
        out_path = f'{BASE_DIR}/{args.out_dir[2:]}'
    elif not args.out_dir.startswith('/'):
        out_path = f'{BASE_DIR}/{args.out_dir}'
    acc = Accelerator(
        project_config=ProjectConfiguration(project_dir=out_path, logging_dir=Path(out_path,args.log_dir)),
        gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with=args.report_to)

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and acc.num_processes > 1:
        raise ValueError("Gradient accumulation is not supported when training the text encoder in distributed training. Please set gradient_accumulation_steps to 1. This feature will be supported in the future.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(acc.state, main_process_only=False)
    if acc.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.use_prior_loss:
        cls_imgs_dir = Path(args.cls_data_dir)
        if not cls_imgs_dir.exists():
            cls_imgs_dir.mkdir(parents=True)
        cur_cls_imgs = len(list(cls_imgs_dir.iterdir()))

        if cur_cls_imgs < args.num_cls_imgs:
            torch_dtype = torch.float16 if acc.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, safety_checker=None,
                                                     torch_dtype=torch_dtype, revision=args.revision)
            pipe.set_progress_bar_config(disable=True)

            num_new_imgs = args.num_cls_imgs - cur_cls_imgs
            logger.info(f"Number of class images to sample: {num_new_imgs}.")

            sample_ds = PromptDataset(args.cls_prompt, num_new_imgs)
            sample_dl = torch.utils.data.DataLoader(sample_ds, batch_size=args.sample_batch_size)
            sample_dl = acc.prepare(sample_dl)
            pipe.to(acc.device)
            for ex in tqdm(sample_dl, disable=not acc.is_local_main_process, desc="Generating class images"):
                imgs = pipe(ex["prompt"]).images
                for i, img in enumerate(imgs):
                    hash_img = hashlib.sha1(img.tobytes()).hexdigest()
                    img.save(cls_imgs_dir / f"{ex['index'][i] + cur_cls_imgs}-{hash_img}.jpg")
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); gc.collect()

    if acc.is_main_process: # Handle the repository creation
        if args.out_dir is not None:
            os.makedirs(args.out_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.out_dir).name,
                                  exist_ok=True, token=args.hub_token).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",
                                                  revision=args.revision, use_fast=False)
    
    # import correct text encoder class
    text_encoder_cls = import_model_cls_from_model_name_or_path(args.pretrained_model_name_or_path,
                                                                args.revision)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    try:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",
                                            revision=args.revision)
    except OSError: # IF does not have a VAE so let's just set it to None (we don't have to error out here)
        vae = None

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                revision=args.revision)

    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if acc.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif acc.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(acc.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(acc.device, dtype=weight_dtype)
    text_encoder.to(acc.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attn_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor,
                                       AttnAddedKVProcessor2_0)):
            lora_attn_processor_cls = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_cls = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor)
        unet_lora_attn_procs[name] = lora_attn_processor_cls(
            hidden_size=hidden_size, cross_attention_dim=cross_attn_dim, rank=args.rank)

    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks. For this,
    # we first load a dummy pipeline with the text encoder and then do the monkey-patching.
    text_encoder_lora_layers = None
    if args.train_text_encoder:
        text_lora_attn_procs = {}
        for name, module in text_encoder.named_modules():
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                text_lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=module.out_proj.out_features, cross_attention_dim=None, rank=args.rank)
        text_encoder_lora_layers = AttnProcsLayers(text_lora_attn_procs)
        temp_pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                      text_encoder=text_encoder)
        temp_pipe._modify_text_encoder(text_lora_attn_procs)
        text_encoder = temp_pipe.text_encoder
        del temp_pipe

    # create custom saving & loading hooks so that `acc.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, out_dir):
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = None

        if args.train_text_encoder:
            text_encoder_keys = acc.unwrap_model(text_encoder_lora_layers).state_dict().keys()
        unet_keys = acc.unwrap_model(unet_lora_layers).state_dict().keys()

        for model in models:
            state_dict = model.state_dict()
            if (text_encoder_lora_layers is not None
                and text_encoder_keys is not None
                and state_dict.keys() == text_encoder_keys
            ): # text encoder
                text_encoder_lora_layers_to_save = state_dict
            elif state_dict.keys() == unet_keys: # unet
                unet_lora_layers_to_save = state_dict
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()
        LoraLoaderMixin.save_lora_weights(out_dir, unet_lora_layers=unet_lora_layers_to_save,
                                          text_encoder_lora_layers=text_encoder_lora_layers_to_save)

    def load_model_hook(models, input_dir):
        # Note we DON'T pass the unet and text encoder here an purpose
        # so that the we don't accidentally override the LoRA layers of
        # unet_lora_layers and text_encoder_lora_layers which are stored in `models`
        # with new torch.nn.Modules / weights. We simply use the pipeline class as
        # an easy way to load the lora checkpoints
        temp_pipe = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype)
        temp_pipe.load_lora_weights(input_dir)

        # load lora weights into models
        models[0].load_state_dict(AttnProcsLayers(temp_pipe.unet.attn_processors).state_dict())
        if len(models) > 1:
            models[1].load_state_dict(AttnProcsLayers(temp_pipe.text_encoder_lora_attn_procs).state_dict())

        # delete temporary pipeline and pop models
        del temp_pipe
        for _ in range(len(models)):
            models.pop()

    acc.register_save_state_pre_hook(save_model_hook); acc.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.lr = args.lr * args.gradient_accumulation_steps * args.train_batch_size * acc.num_processes

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        opt_cls = bnb.optim.AdamW8bit
    else:
        opt_cls = torch.optim.AdamW

    params_to_optimize = ( # Optimizer creation
        itertools.chain(unet_lora_layers.parameters(), text_encoder_lora_layers.parameters())
        if args.train_text_encoder
        else unet_lora_layers.parameters()
    )
    opt = opt_cls(params_to_optimize, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                  weight_decay=args.adam_weight_decay, eps=args.adam_eps)

    if args.precompute_text_embeds:
        def compute_text_embeds(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(
                    tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder, text_inputs.input_ids, text_inputs.attention_mask,
                    text_encoder_use_attn_mask=args.text_encoder_use_attn_mask)
            return prompt_embeds

        precomputed_encoder_hidden_states = compute_text_embeds(args.inst_prompt)
        val_prompt_negative_prompt_embeds = compute_text_embeds("")
        if args.val_prompt is not None:
            val_prompt_encoder_hidden_states = compute_text_embeds(args.val_prompt)
        else:
            val_prompt_encoder_hidden_states = None

        if args.inst_prompt is not None:
            precomputed_inst_prompt_encoder_hidden_states = compute_text_embeds(args.inst_prompt)
        else:
            precomputed_inst_prompt_encoder_hidden_states = None

        text_encoder, tokenizer = None, None
        torch.cuda.empty_cache(); gc.collect()
    else:
        precomputed_encoder_hidden_states = None
        val_prompt_encoder_hidden_states = None
        val_prompt_negative_prompt_embeds = None
        precomputed_inst_prompt_encoder_hidden_states = None

    if any([args.mask_pdm, args.mask_dm, args.mask_prior]): # Setup SAM for mask extraction
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(acc.device)
    else:
        sam_processor, sam_model = None, None

    ds = PDMBoothDataset( # Dataset and DataLoaders creation:
        inst_data_root=args.inst_data_dir, inst_prompt=args.inst_prompt,
        cls_data_root=args.cls_data_dir if args.use_prior_loss else None,
        cls_prompt=args.cls_prompt, cls_num=args.num_cls_imgs, tokenizer=tokenizer,
        center_crop=args.center_crop, tokenizer_max_length=args.tokenizer_max_length,
        encoder_hidden_states=precomputed_encoder_hidden_states, size=args.resolution,
        inst_prompt_encoder_hidden_states=precomputed_inst_prompt_encoder_hidden_states,
        inst_features_root=args.features_data_dir, save_pdm_fs=args.save_pdm_fs,
        sam_processor=sam_processor, sam_model=sam_model, device=acc.device,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dl_num_workers,
        collate_fn=collate_fn,
    )

    # Scheduler and math around the number of training steps.
    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dl) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        override_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=opt, num_cycles=args.lr_num_cycles, power=args.lr_power,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet_lora_layers, text_encoder_lora_layers, opt, dl, lr_scheduler = acc.prepare(
            unet_lora_layers, text_encoder_lora_layers, opt, dl, lr_scheduler)
    else:
        unet_lora_layers, opt, dl, lr_scheduler = acc.prepare(unet_lora_layers, opt, dl, lr_scheduler)

    # recalc total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dl) / args.gradient_accumulation_steps)
    if override_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # recalc number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if acc.is_main_process: # init trackers and store configs
        init_kwargs = {}
        if args.report_to in ('wandb', 'all'):
            init_kwargs['wandb'] = {
                'name': args.wandb_exp if args.wandb_exp else Path(args.out_dir).stem,
                'dir': f'{os.path.dirname(__file__)}/{args.wandb_dir}',
                'settings': wandb.Settings(x_disable_stats=True, x_disable_meta=True,
                                           run_id=wnb_id, x_start_time=wnb_x_start_time),
            }
        trackers_cfg = vars(args)
        test_prompts = trackers_cfg.pop('test_prompts') # `init_trackers` can't handle "list of strings" type
        acc.init_trackers(args.trackers_proj_name, config=trackers_cfg, init_kwargs=init_kwargs)

    if acc.is_main_process:
        for i in range(ds.num_inst_imgs): # images and masks loop
            img = exif_transpose(Image.open(ds.inst_imgs_path[i]))
            img = img.convert("RGB") if not img.mode == "RGB" else img

            if any([args.mask_dm, args.mask_pdm]):
                m_pil = ds.get_mask(img, args.save_pdm_masks, f'masks', f'{i}_mask')
                ds.inst_masks.append(m_pil) if args.mask_dm else ()
                if args.use_pdm_loss and args.mask_pdm:
                    pdm_m_pil = ds.mresize64(ds.ccrop(m_pil))
                    ds.pdm_masks.append(ds.to_tensor(pdm_m_pil)[0].to(acc.device).bool().flatten())
                if not (args.use_prior_loss and args.mask_prior): # del mask processor and model
                    sam_model, sam_processor = None, None
                    torch.cuda.empty_cache(); gc.collect()

            rimg = ds.iresize(img)
            ds.inst_imgs.append(rimg)
            ds.pdm_imgs.append(ds.ccrop(rimg)) if args.use_pdm_loss or test_prompts else ()

    acc.wait_for_everyone() # NOTE: perhaps this line is redundant and can be removed
    if args.use_pdm_loss: # PDM setup
        ddim_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler")
        sd_model = StableDiffusionPipelineWithDDIMInversion(
            vae, text_encoder, tokenizer, unet, ddim_scheduler, safety_checker=None,
            feature_extractor=None,requires_safety_checker=False).to(acc.device)
        for i, pdm_img in tqdm(enumerate(ds.pdm_imgs)):
            if ds.do_extract_pdm_fs: # Extract PDM features (using 'prior' weights)
                pdm_img = ds.img_transforms(pdm_img).to(acc.device, memory_format=torch.contiguous_format,
                                                        dtype=weight_dtype).float().unsqueeze(0)
                Q_fs = extract_pdm_fs(sd_model, pdm_img, args.cls_prompt, acc.device)
                if ds.save_pdm_fs:
                    logger.info(f"Saving PDM features at {ds.inst_features_root}")
                    os.makedirs(ds.inst_features_root, exist_ok=True)
                    torch.save(Q_fs, ds.inst_features_root / f"{ds.inst_imgs_path[i].stem}.pt")
            else: # Load PDM features tensors
                Q_fs = torch.load(ds.inst_features_path[i], map_location=acc.device)
            ds.pdm_fs.append(Q_fs)
        ds.pdm_fs = torch.stack(ds.pdm_fs)
        # Register attn layer of PDM features
        ctrl = AttentionStore()
        attn_layer = unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1
        attn_layer_name = 'up_blocks_3_attentions_2_transformer_blocks_0_attn1_self'
        ptp_utils.register_layer_pdmbooth(attn_layer, attn_layer_name, ctrl)
        l_key = f"Q_{attn_layer_name}"

    # Train!
    total_batch_size = args.train_batch_size * acc.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(ds)}")
    logger.info(f"  Num batches each epoch = {len(dl)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step, first_epoch = 0, 0

    if args.resume_from_ckpt: # Potentially load in the weights and states from a previous save
        if args.use_pdm_loss and ds.do_extract_pdm_fs:
            # Wait until PDM features finished generating (in case they're being generated); to make sure they're being generated using the 'prior' distribution (and not the 'ckpt' distribution).
            acc.wait_for_everyone()
        if args.resume_from_ckpt != "latest":
            path = os.path.basename(args.resume_from_ckpt)
        else: # Get the most recent checkpoint
            dirs = os.listdir(args.out_dir)
            dirs = [d for d in dirs if d.startswith("ckpt")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            acc.print(f"Checkpoint '{args.resume_from_ckpt}' does not exist. Starting a new training run.")
            args.resume_from_ckpt = None
        else:
            acc.print(f"Resuming from checkpoint {path}")
            acc.load_state(os.path.join(args.out_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    prog_bar = tqdm(range(global_step, args.max_train_steps), disable=not acc.is_local_main_process)
    prog_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(dl):
            # Skip steps until we reach the resumed step
            if args.resume_from_ckpt and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    prog_bar.update(1)
                continue

            with acc.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                if vae is not None: # Convert images to latent
                    model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                noise = torch.randn_like(model_input) # Sample noise that we'll add to the latents
                bsz, channels, _, _ = model_input.shape
                timesteps = torch.randint( # Sample a random timestep for each image
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device).long()
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps) # Add noise

                if args.precompute_text_embeds: # Get the text embedding for conditioning
                    encoder_hidden_states = batch["input_ids"]
                else:
                    encoder_hidden_states = encode_prompt(
                        text_encoder, batch["input_ids"], batch["attn_mask"],
                        text_encoder_use_attn_mask=args.text_encoder_use_attn_mask)

                if acc.unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                cls_labels = timesteps if args.cls_labels_conditioning == "timesteps" else None
                model_pred = unet( # Predict the noise residual
                    # NOTE: not performing Classifier-Free Guidance; to align with PDM features extraction
                    noisy_model_input, timesteps, encoder_hidden_states, class_labels=cls_labels).sample

                # if model predicts variance, throw away the prediction. we will only train on the
                # simplified training objective. This means that all schedulers using the fine tuned
                # model must be configured to use one of the fixed variance variance types.
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                inst_img_idx = batch["inst_imgs_indices"][0]
                b_idx = 0
                if any([args.use_inst_loss, args.use_prior_loss]):
                    preds = torch.chunk(model_pred, batch['pixel_values'].shape[0], dim=0)
                    targets = torch.chunk(target, batch['pixel_values'].shape[0], dim=0)
                if args.use_inst_loss: # Instance loss
                    loss = F.mse_loss(preds[b_idx].float(), targets[b_idx].float(), reduction='none')
                    loss = loss[batch["inst_imgs_masks"][0]] if args.mask_dm else loss
                    loss = loss.mean()
                    b_idx += 1
                if args.use_prior_loss: # Prior loss
                    p_loss = F.mse_loss(preds[b_idx].float(), targets[b_idx].float(), reduction='none')
                    p_loss = p_loss[batch["cls_imgs_masks"][0]] if args.mask_prior else p_loss
                    p_loss = p_loss.mean()
                    loss = loss + args.prior_loss_weight * p_loss
                    b_idx += 1
                if args.use_pdm_loss: # PDM loss
                    pred_features = ctrl.attn_store[0][l_key][b_idx].to(acc.device)
                    tgt_features = ds.pdm_fs[inst_img_idx][timesteps[b_idx]]
                    pdm_loss = F.mse_loss(pred_features.float(), tgt_features.float(), reduction='none')
                    pdm_loss = pdm_loss[ds.pdm_masks[inst_img_idx]] if args.mask_pdm else pdm_loss
                    pdm_loss = pdm_loss.mean()
                    loss += args.pdm_loss_weight * pdm_loss
                    ctrl.reset() # reset controller cache for the next iteration

                acc.backward(loss)
                if acc.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_layers.parameters(), text_encoder_lora_layers.parameters())
                        if args.train_text_encoder
                        else unet_lora_layers.parameters()
                    )
                    acc.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                opt.step(); lr_scheduler.step(); opt.zero_grad()

            if acc.sync_gradients: # Checks if accelerator performed an optimization step behind the scenes
                prog_bar.update(1)
                global_step += 1
                if acc.is_main_process:
                    if global_step % args.ckpting_steps == 0:
                        # _before_ saving state, check if this save would set us over the `ckpts_total_limit`
                        if args.ckpts_total_limit is not None:
                            ckpts = os.listdir(args.out_dir)
                            ckpts = [d for d in ckpts if d.startswith("ckpt")]
                            ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
                            # before saving new ckpt, must have at _most_ `ckpts_total_limit - 1` ckpts
                            if len(ckpts) >= args.ckpts_total_limit:
                                num_to_remove = len(ckpts) - args.ckpts_total_limit + 1
                                removing_ckpts = ckpts[0:num_to_remove]
                                logger.info(f"{len(ckpts)} checkpoints already exist, removing {len(removing_ckpts)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_ckpts)}")
                                for r_ckpt in removing_ckpts:
                                    r_ckpt = os.path.join(args.out_dir, r_ckpt)
                                    shutil.rmtree(r_ckpt)

                        save_path = os.path.join(args.out_dir, f"ckpt-{global_step}")
                        acc.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            prog_bar.set_postfix(**logs)
            acc.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

        if acc.is_main_process:
            if args.val_prompt is not None and epoch % args.val_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_val_imgs} images with prompt:"
                    f" {args.val_prompt}.")
                pipe = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path, unet=acc.unwrap_model(unet),
                    text_encoder=None if args.precompute_text_embeds else acc.unwrap_model(text_encoder),
                    revision=args.revision, torch_dtype=weight_dtype)

                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}
                if "variance_type" in pipe.scheduler.config:
                    variance_type = pipe.scheduler.config.variance_type
                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"
                    scheduler_args["variance_type"] = variance_type
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                                         **scheduler_args)

                pipe = pipe.to(acc.device)
                pipe.set_progress_bar_config(disable=True)

                # run inference
                gen = torch.Generator(device=acc.device).manual_seed(args.seed) if args.seed else None
                if args.precompute_text_embeds:
                    pipe_args = {"prompt_embeds": val_prompt_encoder_hidden_states,
                                 "negative_prompt_embeds": val_prompt_negative_prompt_embeds}
                else:
                    pipe_args = {"prompt": args.val_prompt}

                if args.val_imgs is None:
                    imgs = [pipe(**pipe_args, generator=gen).images[0] for _ in range(args.num_val_imgs)]
                else:
                    imgs = []
                    for img in args.val_imgs:
                        imgs.append(pipe(**pipe_args, image=Image.open(img), generator=gen).images[0])

                for t in acc.trackers:
                    if t.name == "tensorboard":
                        np_imgs = np.stack([np.asarray(img) for img in imgs])
                        t.writer.add_images("val", np_imgs, epoch, dataformats="NHWC")
                    if t.name == "wandb":
                        t.log({"val": [wandb.Image(img, caption=f"{i}: {args.val_prompt}")
                                            for i, img in enumerate(imgs)]})

                del pipe; torch.cuda.empty_cache(); gc.collect()

    # Save the lora layers
    acc.wait_for_everyone()
    # finished training so delete SAM (if not deleted yet). NOTE: needed when masking prior loss.
    sam_model, sam_processor = None, None
    torch.cuda.empty_cache(); gc.collect()
    if acc.is_main_process:
        unet = unet.to(torch.float32)
        unet_lora_layers = acc.unwrap_model(unet_lora_layers)

        if text_encoder is not None:
            text_encoder = text_encoder.to(torch.float32)
            text_encoder_lora_layers = acc.unwrap_model(text_encoder_lora_layers)

        LoraLoaderMixin.save_lora_weights(save_directory=args.out_dir, unet_lora_layers=unet_lora_layers,
                                          text_encoder_lora_layers=text_encoder_lora_layers)
        # Load previous pipeline for final inference
        pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision,
                                                 torch_dtype=weight_dtype, safety_checker=None)

        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}
        if "variance_type" in pipe.scheduler.config:
            variance_type = pipe.scheduler.config.variance_type
            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"
            scheduler_args["variance_type"] = variance_type
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, **scheduler_args)
        pipe = pipe.to(acc.device)
        pipe.load_lora_weights(args.out_dir) # load attention processors
        
        # run inference
        hub_imgs = []
        with torch.no_grad():
            if test_prompts and args.num_test_imgs_per_prompt > 0:
                if args.report_to in ('wandb', 'all'):
                    lpips_scores, clip_i_scores, clip_t_scores = [], [], []
                    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
                    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").cuda()
                    clip_model, clip_processor = clip_score.model, clip_score.processor

                    r_imgs = torch.stack([ds.img_transforms(im) for im in ds.pdm_imgs]).to(acc.device).clip(-1,1)
                    scaled_r_imgs = (r_imgs * 127.5 + 127.5).clip(0,255).to(torch.uint8) # [N,C,H,W]

                    proc_r_imgs = clip_processor(text=None, images=scaled_r_imgs, return_tensors="pt", padding=True)
                    r_imgs_fs = clip_model.get_image_features(proc_r_imgs["pixel_values"].to('cuda'))
                    r_imgs_fs /= r_imgs_fs.norm(p=2, dim=-1, keepdim=True)

                gen = torch.Generator(device=acc.device).manual_seed(args.seed) if args.seed else None
                for p in test_prompts:
                    imgs = [pipe(p, num_inference_steps=50, generator=gen).images[0]
                            for _ in range(args.num_test_imgs_per_prompt)]
                    hub_imgs += imgs

                    for t in acc.trackers:
                        if t.name == "tensorboard":
                            np_imgs = np.stack([np.asarray(img) for img in imgs])
                            t.writer.add_images("test", np_imgs, epoch, dataformats="NHWC")
                        if t.name == "wandb":
                            # log images to wandb
                            t.log({"test": [wandb.Image(img, caption=f'{i}: {p}') for i,img in enumerate(imgs)]})
                            # Clac Metrics
                            t_imgs = torch.stack([ds.img_transforms(im) for im in imgs]).to(acc.device).clip(-1,1)
                            scaled_t_imgs = (t_imgs * 127.5 + 127.5).clip(0,255).to(torch.uint8) # [N,C,H,W]
                            # preprocess
                            proc_t_imgs = clip_processor(
                                text=p, images=scaled_t_imgs, return_tensors="pt", padding=True)
                            # get img embeds
                            t_imgs_fs = clip_model.get_image_features(proc_t_imgs["pixel_values"].to('cuda'))
                            t_imgs_fs /= t_imgs_fs.norm(p=2, dim=-1, keepdim=True)
                            # get txt embeds
                            txt_fs = clip_model.get_text_features(
                                proc_t_imgs["input_ids"].to('cuda'), proc_t_imgs["attention_mask"].to('cuda'))
                            txt_fs /= txt_fs.norm(p=2, dim=-1, keepdim=True)

                            # LPIPS (img-img)
                            n_imgs = len(imgs) # num of generated images
                            lpips_s = sum([
                                lpips(t_imgs[i:i+1].repeat(r_imgs.shape[0],1,1,1), r_imgs).item() for i in range(n_imgs)
                            ]) / n_imgs
                            lpips_scores.append(lpips_s)
                            # CLIP-I (img-img)
                            cs_i = torch.matmul(t_imgs_fs, r_imgs_fs.T).clip(-1,1) # [-1,1]
                            scaled_cs_i = cs_i * 0.5 + 0.5 # [0,1]
                            clip_i_scores.append(scaled_cs_i.mean().item())
                            # CLIP-T (txt-img)
                            cs_t = torch.matmul(t_imgs_fs, txt_fs.T).clip(-1,1) # [-1,1]
                            scaled_cs_t = cs_t * 0.5 + 0.5 # [0,1]
                            clip_t_scores.append(scaled_cs_t.mean().item())
                
                if args.report_to in ('wandb', 'all'): # log metrics to wandb
                    del lpips.net, lpips, clip_score, clip_model, clip_processor
                    torch.cuda.empty_cache(); gc.collect()
                    for t in acc.trackers:
                        if t.name == "wandb":
                            t.log({
                                'LPIPS': sum(lpips_scores) / len(lpips_scores),
                                'CLIP_I': sum(clip_i_scores) / len(clip_i_scores),
                                'CLIP_T': sum(clip_t_scores) / len(clip_t_scores),
                            })

        if args.push_to_hub:
            save_model_card(repo_id, imgs=hub_imgs, base_model=args.pretrained_model_name_or_path, pipe=pipe,
                            prompt=args.inst_prompt, train_text_encoder=args.train_text_encoder, repo_folder=args.out_dir)
            upload_folder(repo_id=repo_id, folder_path=args.out_dir, commit_message="End of training",
                          ignore_patterns=["step_*", "epoch_*"])

        if all([args.del_cls_imgs_dir, args.use_prior_loss, args.num_cls_imgs > 0,
                args.cls_data_dir is not None, os.listdir(args.cls_data_dir)]):
            shutil.rmtree(args.cls_data_dir)

    acc.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
