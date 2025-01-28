import os
from numpy import random
random.seed(0)
import argparse
import yaml

OBJECT = 'dog'

BASE_CFG = {
"--out_dir=./weights/temp",

"--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5", // (runwayml/stable-diffusion-v1-5) / (CompVis/stable-diffusion-v1-4) / (stabilityai/stable-diffusion-2-base AND --resolution=768)

"--use_inst_loss", // "--mask_dm",
"--use_pdm", // "--mask_pdm",

"--with_prior_preservation", "--num_cls_imgs=1400", // "--mask_prior",
"--cls_data_dir=./cls_imgs--runwayml_stable-diffusion-v1-5",

"--save_pdm_fs", // "--save_pdm_masks",
"--features_data_dir=./dog_pdm_features--runwayml_stable-diffusion-v1-5",

"--train_text_encoder",
"--max_train_steps=2",
"--lr=1e-4", //"--max_train_steps=500", "--num_train_epochs=100",
//  ---------------------------------------------------------------------------------------------
"--cls_prompt=a photo of dog", "--inst_prompt=a photo of sks dog",
"--inst_data_dir=./dog",
"--train_batch_size=1", "--lr_warmup_steps=0",
"--ckpting_steps=1600", // "--resume_from_ckpt=ckpt_600",
// "--val_prompt=A photo of sks dog in a bucket", "--val_epochs=50",
"--report_to=wandb", // "--wandb_exp=no_name", // "--push_to_hub"
// "--wandb_dir=temp_wand_dir",
"--seed=0",
//  ------------------------------------ List of Test Prompts ------------------------------------
// "--test_prompts=A photo of sks dog in a bucket",
// "--test_prompts=A photo of sks dog sleeping",
// "--test_prompts=A photo of sks dog in the acropolis",
// "--test_prompts=A photo of sks dog swimming",
// "--test_prompts=A photo of sks dog getting a haircut",
}



def run(objects, gpu, camera_sampling, train_dynamic_camera, sample_rand_frames):
    # print(f'gpu: {gpu}')
    for object in objects:
        object_path = object[0]
        rotation = object[1]
        prompt = object[2]
        name = object[3]
        ckpt = object[4]
        token_idx = object[5]
        wandb_exp_name = f'postprocess_attn-mask-token{token_idx}' # TODO change this accordingly
        line = (f'--config custom/configs/animate124-stage2-ms.yaml --train --gpu {gpu} ' +
                f'data.image.image_path={object_path} system.prompt_processor.prompt="{prompt}" ' +
                f'data.image.rotate={rotation} ' +
                'system.loggers.wandb.enable=True ' +
                # 'system.guidance.timestep_strategy=decrease ' +
                f'data.single_view.camera_sampling={camera_sampling} ' +
                f'data.single_view.sample_rand_frames={sample_rand_frames} ' +
                f'system.loggers.wandb.name=stage2_{camera_sampling}_{name}_{wandb_exp_name} ' +
                f'system.prompt_processor.obj_token_clip_idx={token_idx} ' +
                # f'trainer.max_steps=3000 ' +
                f'system.weights={ckpt}'
                )
        print(line)
        cmd = (f'export CUDA_VISIBLE_DEVICES={gpu}; MKL_THREADING_LAYER=GNU; ' +
               f'/cortex/users/orimalca/anaconda3/envs/3d-to-4d/bin/python launch.py {line}')
        os.system(cmd)
        # print(f'cd /cortex/users/orimalca/projects/object_animation && ' +
        #       f'nohup env CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU ' +
        #       f'/cortex/users/orimalca/anaconda3/envs/3d-to-4d/bin/python ' +
        #       f'/cortex/users/orimalca/projects/object_animation/launch.py ' +
        #       f'{line} ' +
        #       f'> /dev/null 2>&1 &'
        #       )
        # print()

def process_batch(gpu, batch, camera_sampling, train_dynamic_camera, sample_rand_frames):
    run(objects=batch, gpu=gpu, camera_sampling=camera_sampling,
        train_dynamic_camera=train_dynamic_camera, sample_rand_frames=sample_rand_frames)

def main_batch(gpus, camera_sampling, train_dynamic_camera, sample_rand_frames):
    file_path = 'custom/scripts/objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    num_gpus = len(gpus)  # Change this to the number of available GPUs
    objects = data['stage2']
    # objects = data['stage2_new_metrics_uniform_incremental']
    batch_size = len(objects) // num_gpus  # Calculate the size of each batch
    batches = [objects[i:i + batch_size] for i in range(0, len(objects), batch_size)]

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, gpu in zip(range(len(batches)), gpus):
            futures.append(executor.submit(process_batch, gpu, batches[i], camera_sampling, train_dynamic_camera, sample_rand_frames))
        
        for future in futures:
            result = future.result()  # If you want to process the result or catch exceptions


if __name__ == '__main__':
    """
    Example to input argument: "--gpus", "0", "1", "4", "5", "6", 
    """
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument('--gpus', type=int, nargs='+', help='List of GPU indices', default=[0])
    parser.add_argument('--camera_sampling', type=str, help='The camera protocol', default="incremental")
    parser.add_argument('--train_dynamic_camera', action='store_true', help='Enable dynamic camera')
    parser.add_argument('--sample_rand_frames', type=str, help='The frame times protocol', default="uniform")
    args = parser.parse_args()

    print('\n')
    main_batch(
        gpus=args.gpus,
        camera_sampling=args.camera_sampling,
        train_dynamic_camera=args.train_dynamic_camera,
        sample_rand_frames=args.sample_rand_frames
        )