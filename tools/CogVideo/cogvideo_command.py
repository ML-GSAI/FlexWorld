import os
import math
import argparse

import torch
import torch.nn.functional as Functional
import numpy as np
from einops import rearrange

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video import SATVideoDiffusionEngineV2V
from arguments import get_args
from sample_video_v2v import get_batch,get_unique_embedder_keys_from_conditioner
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode

class CogVideo:
    def __init__(self, opts ,device='cpu'):
        self.opts = opts.diffusion.cogvideo
        self.device = device
        self.args = None
        self.model = None
        self.image_size = [480, 720]
        self.setup_diffusion()
        
    
    def run_diffusion(self, renderings, prompts=None):
        model,args = self.model, self.args
        sample_func = model.sample
        T, H, W, C, F = args.sampling_num_frames, self.image_size[0], self.image_size[1], args.latent_channels, 8
        
        if prompts is None:
            prompts = self.opts.prompt
        print(f"CogVideo prompts: {prompts}")
        text = prompts
        videos = Functional.interpolate(renderings.permute(0,3,1,2), size=(H, W), mode='bilinear', align_corners=False).permute(0,2,3,1)
        videos = (videos * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(torch.bfloat16)
        
        num_samples = [1]
        force_uc_zero_embeddings = ["txt"]
        device = videos.device
        with torch.no_grad():
            image = videos.contiguous()
            print(image.shape)
            model.first_stage_model = model.first_stage_model.to(device)
            image = model.encode_first_stage(image, None)
            model.first_stage_model = model.first_stage_model.to('cpu')
            image = image.permute(0, 2, 1, 3, 4).contiguous().to(device)
            
            
            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }
            model.to('cuda:0')
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to(device), (c, uc))
                c[k], uc[k] = c[k].to(device), uc[k].to(device)

            c["concat"] = image
            uc["concat"] = image

            for index in range(args.batch_size):
                # reload model on GPU
                model.to(device)
                model.device = device
                model.model = model.model.to(device)
                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                )
                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                # Unload the model from GPU to save GPU memory
                model.to("cpu")
                model.device = "cpu"
                torch.cuda.empty_cache()
                first_stage_model = model.first_stage_model
                first_stage_model = first_stage_model.to(device)

                latent = 1.0 / model.scale_factor * samples_z
                latent = latent.to(device)

                # Decode latent serial to save GPU memory
                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    with torch.no_grad():
                        recon = first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )

                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        return samples

    def nvs_single_view(self, render_results, prompts=None):
        img_H, img_W = render_results.shape[1], render_results.shape[2]
        diffusion_results = self.run_diffusion(render_results,prompts).squeeze(0)
        diffusion_results= Functional.interpolate(diffusion_results, size=(img_H, img_W), mode='bilinear', align_corners=False).permute(0,2,3,1).to(render_results.device)
        return diffusion_results

    def setup_diffusion(self):
        args = get_args([])

        del args.deepspeed_config
        args.model_config.first_stage_config.params.cp_size = 1
        args.model_config.network_config.params.transformer_args.model_parallel_size = 1
        args.model_config.network_config.params.transformer_args.checkpoint_activations = False
        args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
        args.device = "cpu"
        self.args = args
        
        model = get_model(args, SATVideoDiffusionEngineV2V)
        load_checkpoint(model, args)
        model.eval()
        
        self.model=model
        
