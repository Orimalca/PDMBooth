import os
from numpy import random
random.seed(0)
import argparse
from sklearn.model_selection import ParameterGrid
from concurrent.futures import ProcessPoolExecutor

################################ IMPORTANT NOTE Before Running This Script: ################################
# (1) make sure the wandb project (with the desired name) is created before start running as multiple
#     processes will try to create a wandb project with the same name.
# (2) make sure PDM features are created before start running (for the same reason as the previous point).
# (3) make sure parent of checkpoint folder is created (as it will throw an error if not).
############################################################################################################

CORTEX_DIR = '/cortex/users/orimalca'
PROJ_DIR = f'{CORTEX_DIR}/projects/pdmbooth'
OBJECT = 'dog'

CD_PROJ_DIR = f'cd {PROJ_DIR}; '
RUN_LINE = (
    f'{CORTEX_DIR}/anaconda3/envs/pdm/bin/python -m accelerate.commands.launch ' +
    f'{PROJ_DIR}/train_pdmbooth.py '
)

parser = argparse.ArgumentParser(description="Your script description here")
parser.add_argument('--gpus', type=int, nargs='+', default=[0],
    help='List of GPU indices (For example: "--gpus", "0", "1", "4", "5", "6")')
parser.add_argument('--model_id', type=str,
    help='Either "runwayml/stable-diffusion-v1-5" or "CompVis/stable-diffusion-v1-4"')
args = parser.parse_args()

MODEL_ID = args.model_id
MODEL_ID_PARSED = MODEL_ID.replace('/','_')
OBJ_MODEL_DIR_NAME = f'{OBJECT}--{MODEL_ID_PARSED}'
WANDB_PROJECT_NAME = f'PDMBooth_GridSearch-{OBJ_MODEL_DIR_NAME}'
MAX_TRAIN_STEPS = 700
BASE_CFG = (
    f"--lr=1e-6 " +
    f"--train_text_encoder " +
    f"--max_train_steps={MAX_TRAIN_STEPS} " +
    f"--cls_prompt='A photo of {OBJECT}' --inst_prompt='A photo of sks {OBJECT}' " +
    f"--inst_data_dir=./{OBJECT} " +
    f"--train_batch_size=1 --lr_warmup_steps=0 --ckpting_steps=1600 " +
    f"--report_to=wandb --seed=0 " +
    # model dependent arguments
    f"--pretrained_model_name_or_path={MODEL_ID} " +
    f"--cls_data_dir=./cls_imgs/{OBJ_MODEL_DIR_NAME} " +
    f"--features_data_dir=./pdm_features/{OBJ_MODEL_DIR_NAME} " +
    f"--trackers_proj_name={WANDB_PROJECT_NAME} " +
    f"--out_dir=./ckpts/{WANDB_PROJECT_NAME}/exp " +
    # losses
    f"--use_inst_loss " +
    # f"--use_pdm " +
    # f"--pdm_loss_weight=0.05 " +
    f"--use_prior_loss --num_cls_imgs={MAX_TRAIN_STEPS} " +
    # f"--use_prior_loss --num_cls_imgs=1000 " +
    # f"--mask_pdm --mask_dm --mask_prior " +
    # test prompts
    f"--test_prompts='A photo of sks {OBJECT} in a bucket' " +
    f"--test_prompts='A photo of sks {OBJECT} sleeping' " +
    f"--test_prompts='A photo of sks {OBJECT} in the acropolis' " +
    f"--test_prompts='A photo of sks {OBJECT} swimming' " +
    f"--test_prompts='A photo of sks {OBJECT} getting a haircut' "
)



def run(combs, gpu):
    GPU_LINE = f'export CUDA_VISIBLE_DEVICES={gpu}; MKL_THREADING_LAYER=GNU; '
    
    for comb in combs:
        COMB_PARAMS = ''
        for p,v in comb.items():
            if isinstance(v, bool):
                param_to_add = f"--{p} " if v else ""
            elif isinstance(v, str):
                param_to_add = f"--{p}='{v}' "
                ...
            elif isinstance(v, (int, float)):
                param_to_add = f"--{p}={v} "
            else:
                param_to_add = ""
                raise ValueError()
            
            COMB_PARAMS += param_to_add

        cmd = (
            CD_PROJ_DIR +
            GPU_LINE +
            RUN_LINE +
            BASE_CFG + COMB_PARAMS
        )

        # print(cmd)
        os.system(cmd)


def process_batch(gpu, batch):
    run(combs=batch, gpu=gpu)


def main_batch(gpus, combs=[dict()]):
    total_num_runs = len(combs)
    
    num_gpus = len(gpus) # number of available GPUs
    if total_num_runs < num_gpus: # in case number of available GPUs is greater than total number of runs
        gpus = gpus[:total_num_runs] # use only the first total_num_runs GPUs
    num_gpus = len(gpus) # number of GPUs we'll use
    batch_size = total_num_runs // num_gpus # Calculate the size of each batch

    batches_gpus = [combs[i:i+batch_size] for i in range(0, total_num_runs, batch_size)]
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, gpu in zip(range(len(batches_gpus)), gpus):
            futures.append(executor.submit(process_batch, gpu, batches_gpus[i]))
        
        for future in futures:
            result = future.result() # save `result` if you want to process the result or catch exceptions


if __name__ == '__main__':
    ############################################ OPTION 1 ############################################
    # PARAMS_GRID = {
    #     'train_text_encoder': [True, False],
    #     'use_inst_loss': [True, False], 'use_pdm': [True, False], 'use_prior_loss': [True, False],
    #     'mask_dm': [True, False], 'mask_pdm': [True, False], 'mask_prior': [True, False],
    # }
    # filters = [
    #     lambda c: not (c['use_prior_loss'] and not c['use_inst_loss'] and not c['use_pdm']),
    #     lambda c: not (not c['use_prior_loss'] and c['mask_prior']),
    #     lambda c: not (not c['use_inst_loss'] and c['mask_dm']),
    #     lambda c: not (not c['use_pdm'] and c['mask_pdm']),
    #     lambda c: c['use_pdm'],
    #     lambda c: c['use_inst_loss'],
    # ]

    ############################################ OPTION 2 ############################################
    # NOTE: Option 2 gives 48 combinations/runs, so choose number of gpus that 48 is divided by (e.g., 3)
    
    PARAMS_GRID = {
        # NOTE:
        # (1) use_pdm and use_inst_loss are always True
        # (2) model_id is given as a parameter to this script
        # (3) train_text_encoder is always False since training the text encoder always cause overfitting
        'mask_dm': [True, False], # 'mask_pdm': [True, False],
        'mask_prior': [True, False],
        # 'pdm_loss_weight': [0.05, 0.1],
        # 'max_train_steps': [700, 800, 900, 1000]
        # 'lr': [0.000001, 0.0000008]
    }
    # filters = [
    #     # lambda c: not (not c['use_prior_loss'] and c['mask_prior']),
    #     lambda c: not (not c['mask_dm'] and not c['mask_prior']),
    # ]

    combs = list(ParameterGrid(PARAMS_GRID)) # Generate all combinations
    # for filter_f in filters: # Apply filters iteratively
    #     combs = list(filter(filter_f, combs))

    # for c in combs: # Print or process the filtered combinations
    #     print(c)
    # print(f'A total of {len(combs)} combinations.')

    main_batch(combs=combs, gpus=args.gpus)