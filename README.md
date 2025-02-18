# PDMBooth

PDMBooth is a method to personalize T2I Diffusion Models given just a few(3~5) images of a subject.
The `train_pdmbooth.py` script shows how to implement the training procedure and adapt it for stable diffusion.

>**Abstract**: <br>
> Recent advances in T2I models have enabled high-quality personalized image synthesis of user-provided concepts with flexible textual control. In this work, we analyze the limitations of DreamBooth (DB), a primary technique in T2I personalization. As well as providing an alternative that can tackle those limitations. When integrating the learned concept into new prompts, DB tends to overfit the concept, which can be attributed to its coarse objective. We introduce PDMBooth, a novel approach that addresses this issue by integrating a semantic regularization term. Furthermore, we try experimenting with masking DB's objective. Compared to DB, our method demonstrates significant improvements in text alignment and similar results in identity preservation.

<img width="1010" alt="Screenshot 2025-02-17 at 19 13 25" src="https://github.com/user-attachments/assets/1cda8d1b-f579-4890-8e81-a18aaa217a8d" />

## Environment Setup & Installation

For installation please have `conda` properly installed and then run the following commands:

<ins>NOTE</ins>: use CUDA 11.8

```bash
conda create -y -n pdm python=3.11.10
conda install --yes pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

<ins>**Important Note**</ins>: In case of encountering ```ImportError: cannot import name 'cached_download' from 'huggingface_hub' (<path_to_your_conda_env>/lib/python3.11/site-packages/huggingface_hub/__init__.py)```, that's a [well known bug](https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2425265522) in the `diffusers` library. To fix, remove `cached_download` from the import line in `<path_to_your_conda_env>/lib/python3.11/site-packages/diffusers/utils/dynamic_modules_utils.py` as explain [here](https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2425265522). In short:

```python
# BEFORE the fix it should look as follows:
from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info
# AFTER the fix it should look like this:
from huggingface_hub import HfFolder, hf_hub_download, model_info
```

After that, initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with one of the following options:

```bash
### (OPTION 1): init accelerate and choose the specific configurations you want
accelerate config # NOTE: specifying `torch compile mode` to True can cause dramatic speedups

### (OPTION 2): default accelerate configuration (without answering questions)
accelerate config default

### (OPTION 3): if your env doesn't support an interactive shell (e.g., a notebook)
from accelerate.utils import write_basic_config
write_basic_config()
```

## Training

Let's get our reference subject. We'll use the dog images from [this](https://huggingface.co/datasets/diffusers/dog-example) link. To download them locally you can use the following script:

```python
from huggingface_hub import snapshot_download
snapshot_download("diffusers/dog-example", local_dir="./dog",
                  repo_type="dataset", ignore_patterns=".gitattributes")
```

Now, launch the training using `train_pdmbooth.py` as follows:

```bash
accelerate launch train_pdmbooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --inst_data_dir=./dog \
  --out_dir="ckpts/path-to-save-model" \
  --inst_prompt="a photo of sks dog" --cls_prompt="a photo of dog" \
  --lr=1e-6 --max_train_steps=700 --ckpting_steps=1400 --train_text_encoder \
  --train_batch_size=1 --lr_warmup_steps=0 \
  --resolution=512 \
  --cls_data_dir="path-to-class-images" \
  --use_inst_loss --use_prior_loss --use_pdm --mask_pdm --mask_dm \
  --num_cls_imgs=700 \
  --report_to=wandb \
  --test_prompts="A photo of sks dog in a bucket" \
  --test_prompts="A photo of sks dog sleeping" \
  --test_prompts="A photo of sks dog in the acropolis" \
  --test_prompts="A photo of sks dog swimming" \
  --test_prompts="A photo of sks dog getting a haircut" \
  --seed=0
```

<ins>NOTES</ins>:
- We use [`wandb`](https://docs.wandb.ai/quickstart) to monitor training progress. If you want to disable it, simply remove the `report_to` and `test_prompts` arguments. We automatically plot the `test_prompts` in `wandb` framework. If you decide to not use `wandb` then you have to run inference afterwards, which means you'll need to save the model weights so add the `--save_model_weights` argument and then run the provided inference script under the inference section.
- It's recommended to generate `max_train_steps` images for the prior-preservation objective. The `num_cls_imgs` flag sets the number of images to generate with the class prompt. You can also place existing images in `cls_data_dir`, and the training script will generate any additional images.
- The script also fine-tunes the `text_encoder` along with the `unet`. Training the text encoder requires more memory, with this option the training won't fit on 16GB GPU. It needs at least 24GB VRAM. To disable this, simply don't pass the `--train_text_encoder` argument to the script.
- Here we are using [Stable Diffusion 1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Change the `resolution` to 768 if you want to use [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.
- To enable xFormers for memory efficient attention, run `pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118` and add the `--enable_xformers_memory_efficient_attention` argument.

### Training with Low-Rank Adaptation (LoRA)

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*

In a nutshell, LoRA allows to adapt pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:
- Previous pretrained weights are kept frozen so that the model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted towards new training images via a `scale` parameter.

To run with LoRA use the `train_pdmbooth_lora.py` script as follows:

```bash
accelerate launch train_pdmbooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --inst_data_dir=./dog \
  --out_dir="ckpts/path-to-save-model" \
  --inst_prompt="a photo of sks dog" --cls_prompt="a photo of dog" \
  --resolution=512 \
  --use_inst_loss --use_pdm --mask_pdm --mask_dm \
  --lr=1e-4 --max_train_steps=700 --ckpting_steps=1400 \
  --train_batch_size=1 --lr_warmup_steps=0 \
  --report_to=wandb \
  --test_prompts="A photo of sks dog in a bucket" \
  --test_prompts="A photo of sks dog sleeping" \
  --test_prompts="A photo of sks dog in the acropolis" \
  --test_prompts="A photo of sks dog swimming" \
  --test_prompts="A photo of sks dog getting a haircut" \
  --seed=0
```

<ins>NOTES</ins>:
- The final LoRA embedding weights ~3MB in size which is orders of magnitudes smaller than the original model. Thus, when using LoRA we save the model weights by default, so no need to add the `--save_model_weights` argument.
- When using LoRA we can use a much higher learning rate. Here we use *1e-4* instead of the usual *1e-6*.
- Optionally, we can also train additional LoRA layers for the text encoder. Specify the `--train_text_encoder` argument above for that.
- When using LoRA we don't utilize our prior preservation objective, so no need to pass arguments related to it.

## Inference

Once you have trained a model, you can run inference simply by using the provided inference script `infer_pdmbooth.py` as follows:

```bash
python infer_pdmbooth.py \
  "0" \ # seed
  "dog" \ # object class
  "relative_path_to_model" \ # path to model weights without 'ckpts/' parent
  "false" \ # whether trained a LoRA or not
  "runwayml/stable-diffusion-v1-5" \ # model id
  "false" \ # whether trained text_encoder as well when trained a LoRA
```

## Running on the Dataset

We also provide scripts to reproduce the quantitative results of our method on DreamBooth's dataset.

<ins>NOTES</ins>: Make sure to check the following points before start running
- Use the same python envrionment used for `train_pdmbooth.py`.
- A `wandb` project with the name `PDMBooth-dreambooth-ds` is exists (for LoRA use `PDMBooth-lora-dreambooth-ds`).
- A directory with the name `/ckpts/PDMBooth-dreambooth-ds` exists (for LoRA use `/ckpts/PDMBooth-lora-dreambooth-ds`).
- For each class in the dataset, the class images dir exists under the path `cls_imgs/object_class_name` (e.g., `cls_imgs/dog`). When using LoRA we can skip this step because we don't utilize our prior preservation objective when using LoRA.

```bash
conda activate pdm
python scripts/run_all.py "--gpus", "6", "7" # GPUs to use when running on the dataset
```
<ins>NOTE</ins>: For running our method on the dataset with LoRA, use `scripts/run_all_lora.py` instead.