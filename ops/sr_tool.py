import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/Real_ESRGAN'
sys.path.insert(0,reference)

from ops.utils.general import convert_PIL
import numpy as np
import torch

class SRTool():
    def __init__(self,device='cuda'):
        from realesrgan_command import RealESRGAN
        
        self.upsampler = RealESRGAN(device=device)
        
    def __call__(self, input_image, upscale=4.0, downscale=1.0):
        """
            input_image: [H, W, 3] or [B, H, W, 3], range in [0, 1]
            upscale: float, usually > 1.0,  the upscale factor for super resolution
            downscale: float, usually <= 1.0, the downscale factor to resize the input image
            return: np.ndarray
        """
        if isinstance(input_image, np.ndarray):
            input_image = torch.from_numpy(input_image)
            
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
            
        for i in range(input_image.size(0)):
            image = convert_PIL(input_image[i])
            if downscale != 1.0:
                w, h = image.size
                image = image.resize((int(w * downscale), int(h * downscale)))
            image = np.array(image)
            output = self.upsampler(image, outscale = upscale)
            output = torch.from_numpy(output / 255.0).unsqueeze(0)
            if i == 0:
                outputs = output
            else:
                outputs = torch.cat([outputs, output], 0)
                
        if outputs.size(0) == 1:
            outputs = outputs[0]
                
        return outputs
        
# import torch
# from diffusers import FluxControlNetModel
# from diffusers.pipelines import FluxControlNetPipeline
# from torchvision import transforms
# from ops.utils.general import convert_PIL

# class SRTool():
#     def __init__(self,device='cuda'):
#         torch.cuda.empty_cache()
#         # Load pipeline
#         controlnet = FluxControlNetModel.from_pretrained(
#         "jasperai/Flux.1-dev-Controlnet-Upscaler",
#         torch_dtype=torch.bfloat16
#         )
#         pipe = FluxControlNetPipeline.from_pretrained(
#         "black-forest-labs/FLUX.1-dev",
#         controlnet=controlnet,
#         torch_dtype=torch.bfloat16
#         )
#         pipe.to(device)
#         self.pipe = pipe
    
#     def __call__(self, image, upscale=1.0):
#         """
#             return: torch.Tensor
#         """
#         # Load a control image
#         control_image = convert_PIL(image)
#         if upscale != 1.0:
#             w, h = control_image.size
#             control_image = control_image.resize((int(w * upscale), int(h * upscale)))
            
#         image = self.pipe(
#             prompt="", 
#             control_image=control_image,
#             controlnet_conditioning_scale=0.6,
#             num_inference_steps=28, 
#             guidance_scale=3.5,
#             height=control_image.size[1],
#             width=control_image.size[0]
#         ).images[0]
#         image = transforms.ToTensor()(image)
#         return image


