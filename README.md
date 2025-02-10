# PDMBooth training example

PDMBooth is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
The `train_pdmbooth.py` script shows how to implement the training procedure and adapt it for stable diffusion.


## Running locally with PyTorch

### Installation

<u>NOTE</u>: Please use CUDA 11.8

```bash
conda create -y -n pdm python=3.11.10
conda install --yes pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

If you tackle the following error:
```bash
ImportError: cannot import name 'cached_download' from 'huggingface_hub' (/cortex/users/orimalca/anaconda3/envs/pdm_temp1/lib/python3.11/site-packages/huggingface_hub/__init__.py)
```

That's a [well known bug](https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2425265522) in the `diffusers` library. To fix, remove `cached_download` from the import line in `<path_to_conda_env>/lib/python3.11/site-packages/diffusers/utils/dynamic_modules_utils.py` as explain [here](https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2425265522).

Before the fix it should look as follows:
```python
from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info
```

After the fix it should look as follows:
```python
from huggingface_hub import HfFolder, hf_hub_download, model_info
```

After that, initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups. 

### Dog example

Now let's get our dataset. For this example we will use the dog images from [this](https://huggingface.co/datasets/diffusers/dog-example) link:

Let's first download the example images locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

And launch the training using: (here we'll use [Stable Diffusion 1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5))

```bash
accelerate launch train_pdmbooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --inst_data_dir=./dog \
  --out_dir="path-to-save-model" \
  --inst_prompt="a photo of sks dog" \
  --cls_prompt="a photo of dog" \
  --lr=1e-6 --max_train_steps=700 \
  --train_batch_size=1 \
  --lr_warmup_steps=0 \
  --resolution=512 \
  --seed=0
```

<u>Note</u>: Change the `resolution` to 768 if you want to use [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to DreamBooth's paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to DreamBooth's paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. The `num_cls_imgs` flag sets the number of images to generate with the class prompt. You can place existing images in `cls_data_dir`, and the training script will generate any additional images so that `num_cls_imgs` are present in `cls_data_dir` during training time.

```bash
accelerate launch train_pdmbooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --inst_data_dir=./dog \
  --cls_data_dir="path-to-class-images" \
  --out_dir="path-to-save-model" \
  --use_prior_loss \
  --inst_prompt="a photo of sks dog" \
  --cls_prompt="a photo of dog" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --lr=1e-6 \
  --lr_warmup_steps=0 \
  --num_cls_imgs=700 \
  --max_train_steps=700 \
  --resolution=512 \
  --seed=0
```

### Fine-tune text encoder with the UNet.

The script also allows to fine-tune the `text_encoder` along with the `unet`. It's been observed experimentally that fine-tuning `text_encoder` gives much better results (especially on faces). 
Pass the `--train_text_encoder` argument to the script to enable training the text encoder.

<u>Note</u>: Training the text encoder requires more memory, with this option the training won't fit on 16GB GPU. It needs at least 24GB VRAM.

```bash
accelerate launch train_pdmbooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_text_encoder \
  --inst_data_dir=./dog \
  --cls_data_dir="path-to-class-images" \
  --out_dir="path-to-save-model" \
  --use_prior_loss \
  --inst_prompt="a photo of sks dog" \
  --cls_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr=1e-6 \
  --lr_warmup_steps=0 \
  --num_cls_imgs=700 \
  --max_train_steps=700 \
  --seed=0
```

### Inference

Once you have trained a model using the above command, you can run inference simply using the provided inference file as follows:

```bash
python infer_pdmbooth.py \
  "0" \ # seed
  "dog" \ # object class
  "./temp" \ # path to saved model weights dir
  "true" \ # whether using LoRA or not
  "runwayml/stable-diffusion-v1-5" \ # model id
  "true" \ # whether trained text_encoder as well when trained LORA
```

### Inference from a training checkpoint

You can also perform inference from one of the checkpoints saved during the training process, if you used the `--ckpting_steps` argument. Please, refer to [DreamBooth's documentation](https://huggingface.co/docs/diffusers/v0.18.2/en/training/dreambooth#inference-from-a-saved-checkpoint) to see how to do it.

## Training with Low-Rank Adaptation of Large Language Models (LoRA)

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*

In a nutshell, LoRA allows to adapt pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:
- Previous pretrained weights are kept frozen so that the model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted towards new training images via a `scale` parameter.

### Training

Let's get started with a simple example. We'll re-use the dog example of the [previous section](#dog-example).

We'll still use [Stable Diffusion 1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

**___Note: It is quite useful to monitor the training progress by regularly generating sample images during training. [wandb](https://docs.wandb.ai/quickstart) is a nice solution to easily see generated images during training. All you need to do is to run `pip install wandb` before training and pass `--report_to="wandb"` to automatically log images.___**

```bash
accelerate launch train_pdmbooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --inst_data_dir=./dog \
  --out_dir="path-to-save-model" \
  --inst_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --ckpting_steps=100 \
  --lr=1e-4 \
  --report_to="wandb" \
  --lr_warmup_steps=0 \
  --max_train_steps=700 \
  --seed=0
```

**<u>Notes</u>**:
- When using LoRA we can use a much higher learning rate compared to vanilla pdmbooth. Here we use *1e-4* instead of the usual *1e-6*.
- The final LoRA embedding weights ~3MB in size which is orders of magnitudes smaller than the original model.
- Optionally, we can also train additional LoRA layers for the text encoder. Specify the `--train_text_encoder` argument above for that.
- With the default hyperparameters from the above, the training seems to go in a positive direction.


### Enable xFormers
To enable xFormers for memory efficient attention, add `--enable_xformers_memory_efficient_attention` argument and run:
```bash
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```


### Inference

After training, LoRA weights can be loaded very easily into the original pipeline. First, you need to 
load the original pipeline. Follow the inference script to see how.



