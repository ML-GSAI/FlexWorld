import sys
import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2  
import glob
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from pathlib import Path
from torchvision.utils import save_image

class ViewCrafter:
    def __init__(self, opts,device='cpu'):
        self.opts = opts.diffusion.viewcrafter
        self.device = device
        self.setup_diffusion()
    
    def run_diffusion(self, renderings, prompts=None):
        """
            renderings: video tensor
            prompts: str
        """
        if prompts is None:
            prompts = [self.opts.prompt]
        else:
            prompts = [prompts]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            # [1,1,c,t,h,w]
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index)

            # save_results_seperate(batch_samples[0], self.opts.save_dir, fps=8)
            # torch.Size([1, 3, 25, 576, 1024]) [-1,1]

        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.) 

    def nvs_single_view(self, render_results,prompts=None):
        diffusion_results = self.run_diffusion(render_results,prompts=prompts)
        diffusion_results = (diffusion_results + 1.0) / 2.0
        return diffusion_results

    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]
