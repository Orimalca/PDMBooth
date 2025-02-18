import os
from sys import executable as PATH_TO_PYTHON_ENV
import yaml
from numpy import random
random.seed(0)
import argparse
from concurrent.futures import ProcessPoolExecutor

################################ IMPORTANT NOTE Before Running This Script: ################################
# (1) make sure the wandb project (with the desired name) is created before start running, as multiple
#     processes will try to create a wandb project with the same name and it will throw and error.
# (2) make sure parent of checkpoint folder is created (as it will throw an error if not).
############################################################################################################

PROJ_DIR = os.path.dirname(__file__)[:-len('/scripts')]
DATASET_DIR = f'{PROJ_DIR}/dataset'

CD_PROJ_DIR = f'cd {PROJ_DIR}; '
RUN_LINE = (
    f'{PATH_TO_PYTHON_ENV} -m accelerate.commands.launch ' +
    f'{PROJ_DIR}/train_pdmbooth_lora.py '
)

parser = argparse.ArgumentParser(description="Script for running on dataset")
parser.add_argument('--gpus', type=int, nargs='+', default=[0],
    help='List of GPU indices (For example: "--gpus", "0", "1", "4", "5", "6")')
parser.add_argument('--unique_token', type=str, default='sks', help='The unique identifier to use')
parser.add_argument('--path_to_yaml', type=str, default=f'{PROJ_DIR}/scripts/objects.yaml',
    help='Path to YAML containing the objects, live subjects, objects prompts, and live subjects prompts')
args = parser.parse_args()

with open(args.path_to_yaml, 'r') as file:
    data = yaml.safe_load(file)
OBJECTS = data['objects']
OBJECTS_PROMPTS = [l[0] for l in data['objects_prompts']]
LIVE_SUBJECTS_PROMPTS = [l[0] for l in data['live_subjects_prompts']]
UNIQUE_TOKEN = args.unique_token
WANDB_PROJECT_NAME = f'PDMBooth-lora-dreambooth-ds'
BASE_CFG = (
    f"--lr=1e-4 --max_train_steps=700 " +
    f"--train_batch_size=1 --lr_warmup_steps=0 --ckpting_steps=1600 " +
    f"--report_to=wandb --seed=0 " +
    f"--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 " +
    f"--trackers_proj_name={WANDB_PROJECT_NAME} " +
    f"--use_pdm_loss " +
    f"--mask_pdm --mask_dm " +
    f"--use_inst_loss " # not using prior preservation when using LoRA
)


def run(objects, gpu):
    GPU_LINE = f'export CUDA_VISIBLE_DEVICES={gpu}; export MKL_THREADING_LAYER=GNU; '

    for obj in objects:
        subject_name = obj[0]
        subject_class = obj[1]
        
        OBJECT_PARAMS = (
            f"--inst_prompt='A photo of {UNIQUE_TOKEN} {subject_class}' " +
            f"--cls_prompt='A photo of {subject_class}' " +
            f"--inst_data_dir={DATASET_DIR}/{subject_name} " +
            f"--out_dir=./ckpts/{WANDB_PROJECT_NAME}/{subject_name} "
        )
        
        test_prompts = OBJECTS_PROMPTS if obj[2] == '0' else LIVE_SUBJECTS_PROMPTS
        for prompt in test_prompts:
            OBJECT_PARAMS += f"--test_prompts='{prompt.format(UNIQUE_TOKEN, subject_class)}' "
        
        cmd = (
            CD_PROJ_DIR +
            GPU_LINE +
            RUN_LINE +
            BASE_CFG + OBJECT_PARAMS
        )

        # print(cmd)
        os.system(cmd)


def process_batch(gpu, batch):
    run(objects=batch, gpu=gpu)


if __name__ == '__main__':
    total_num_runs = len(OBJECTS)
    num_gpus = len(args.gpus) # number of available GPUs
    batch_size = total_num_runs // num_gpus # Calculate the size of each batch
    batches_gpus = [OBJECTS[i:i+batch_size] for i in range(0, total_num_runs, batch_size)]
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, gpu in zip(range(len(batches_gpus)), args.gpus):
            futures.append(executor.submit(process_batch, gpu, batches_gpus[i]))

        for future in futures:
            result = future.result() # save `result` if you want to process the result or catch exceptions
