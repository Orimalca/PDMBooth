import PIL
import numpy as np
import torch
from diffusers import DDIMScheduler
from matplotlib import pyplot as plt
from torch import nn
from transformers import SamModel, SamProcessor
from attention_store import AttentionStore
from StableDiffusionPipelineWithDDIMInversion import StableDiffusionPipelineWithDDIMInversion
#from sam_utils import show_masks_on_image, show_points_on_image
import ptp_utils


def center_crop(im):
    width, height = im.size  # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_into_format_from_path(im_path):
    return center_crop(PIL.Image.open(im_path)).resize((512, 512))

def get_masks(raw_image, input_points):
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    # visualize sam segmentation
    # scores = outputs.iou_scores
    # show_masks_on_image(raw_image, masks[0], scores)
    # show_points_on_image(raw_image, input_points[0])
    return masks


def extract_attention(sd_model, img, prompt, img_rel_path, device='cuda'):
    ctrl = AttentionStore()
    res_attn = 64
    
    ## NOTE: without inversion
    # latents, _ = sd_model.get_latent(prompt, image=img, deterministic=True)
    # ptp_utils.register_attention_control_efficient(sd_model, ctrl)
    # _ = sd_model(prompt, latents=latents, guidance_scale=1.0, output_type="latent",
    #              num_inference_steps=1, fake_clean=True, fake_t=999)
    # l_name = 'up_blocks_3_attentions_2_transformer_blocks_0_attn1_self'
    # i = 0
    # return [ctrl.attn_store[i][f'Q_{l_name}'][0].to(device),
    #         ctrl.attn_store[i][f'K_{l_name}'][0].to(device),
    #         ], sd_model.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1, res_attn
    
    ## NOTE: with inversion
    steps = 50 # use 999 to get more precise instance features (NOTE very minor; 50 steps is enough; especially for the inversion; for better quality you can use 999 steps in the reconstruction)
    inv_latents, _ = sd_model.invert(prompt, image=img, num_inference_steps=steps,
                                     guidance_scale=1.0, deterministic=True)
    ptp_utils.register_attention_control_efficient(sd_model, ctrl)
    recon_img = sd_model(prompt, latents=inv_latents, guidance_scale=1.0, num_inference_steps=steps).images[0]
    recon_img.save(f"{imgs_dir}/recon_{img_rel_path}", 'PNG')
    l_n = 'up_blocks_3_attentions_2_transformer_blocks_0_attn1_self' # layer name
    i = steps - 1 # features of step t=0 (NOTE: i=(recon_steps-1) corresponds to t=0 and i=0 to t=recon_steps)
    return [ctrl.attn_store[i][f'Q_{l_n}'][0].to(device), ctrl.attn_store[i][f'K_{l_n}'][0].to(device),
            ], sd_model.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1, res_attn

def heatmap(sd_model, ref_img, tgt_img):
    image_size = 512
    prompt = f"A photo" # A photo of dog" or "A photo of sks dog" does not yield inferior features (NOTE even some nonsense prompt "blablabla asgldkslds" gives the same result/features)
    # extract query PDM features (Q and K)
    ref_attn_ls, _, res_attn = extract_attention(sd_model, ref_img, prompt, ref_rel_path)
    h = w = res_attn

    # get query mask using SAM or use provided mask from user
    source_masks = get_masks(ref_img.resize(size=(h,w)), [[[h//2, w//2]]])
    source_mask = source_masks[0][:,1:2,:,:].squeeze(dim=0).squeeze(dim=0)
    mask_idx_y, mask_idx_x = torch.where(source_mask)

    # extract target PDM features (Q and K)
    target_attn_ls, _, _ = extract_attention(sd_model, tgt_img, prompt, tgt_rel_path)

    # apply matching and show heatmap
    for attn_idx, (ref_attn, target_attn) in enumerate(zip(ref_attn_ls, target_attn_ls)):
        heatmap = torch.zeros(ref_attn.shape[0]).to("cuda")
        for x, y in zip(mask_idx_x, mask_idx_y):
            t = np.ravel_multi_index((y,x), dims=(h,w))
            source_vec = ref_attn[t].reshape(1,-1)
            euclidean_dist = torch.cdist(source_vec, target_attn)
            idx = torch.sort(euclidean_dist)[1][0][:100]
            heatmap[idx] += 1
        heatmap = heatmap / heatmap.max()
        heatmap_img_size = \
            nn.Upsample(size=(image_size,image_size), mode='bilinear')(heatmap.reshape(1,1,64,64))[0][0]
        plt.imshow(tgt_img)
        plt.imshow(heatmap_img_size.cpu(), alpha=0.6)
        plt.show()


if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    imgs_dir = 'dogs'
    ref_rel_path = 'source.png'
    tgt_rel_path = 'target.png'

    sd_model = StableDiffusionPipelineWithDDIMInversion.from_pretrained(
        model_id,
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    )
    sd_model = sd_model.to(device)
    img_size = 512

    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    ref_img = load_im_into_format_from_path(f"{imgs_dir}/{ref_rel_path}").convert("RGB")
    plt.imshow(ref_img)
    plt.show()
    tgt_img = load_im_into_format_from_path(f"{imgs_dir}/{tgt_rel_path}").convert("RGB")
    plt.imshow(tgt_img)
    plt.show()
    heatmap(sd_model, ref_img, tgt_img)

