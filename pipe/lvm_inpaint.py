'''
render using frames in GS
inpaint with fooocus
'''
import torch
import numpy as np
from ops.llava import Llava
from ops.flux import FLUX
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation
from ops.utils.general import convert_PIL

def expand_and_mask_image(input_image:Image.Image,size=(1024,576),scale=None):
    W ,H = size
    # input_image = Image.open(input_image_path)
    w1, h1 = input_image.size
    
    if scale is None:
        scale = min(W, H) / min(w1, h1)
        
    new_w = int(w1 * scale)
    new_h = int(h1 * scale)

    resized_image = input_image.resize((new_w, new_h), Image.LANCZOS)

    target_width, target_height = W, H
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))

    left = (target_width - new_w) // 2
    top = (target_height - new_h) // 2

    new_image.paste(resized_image, (left, top))

    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    mask[top:top+new_h, left:left+new_w] = 255
    
    mask_image = Image.fromarray(mask).convert('L')
    mask_image=ImageOps.invert(mask_image)
    
    new_image.save('./cache/resized_image_rgb.png')
    mask_image.save('./cache/resized_image_mask.png')
    return input_image,new_image, mask_image

def resize_and_center(image_np,size=(512,512)):
    image = Image.fromarray(image_np.astype(np.uint8))

    new_size = (int(size[0] * 0.8), int(size[1] * 0.8))
    resized_image = image.resize(new_size, Image.LANCZOS)

    new_image = Image.new("RGBA", size, (0, 0, 0, 0))

    top_left_x = (size[0] - new_size[0]) // 2
    top_left_y = (size[1] - new_size[1]) // 2

    new_image.paste(resized_image, (top_left_x, top_left_y))
    
    
    new_image.save('./cache/centered_image_rgba.png')
    
    new_image=np.array(new_image)
    mask = new_image[..., 3] / 255.0
    return new_image, mask

class Inpaint_Tool():
    def __init__(self,opt) -> None:
        self.opt = opt
        self.size = tuple(opt.size)
        self.generator = torch.Generator(device="cuda").manual_seed(opt.seed)
        self._load_model()
        
    def _load_model(self):
        self.flux = FLUX(self.opt)
        self.llava = None
        
    def get_prompt(self, image:Image.Image, prompt=None, llava_prompt=None):
        """
            if prompt is None, use llava to generate prompt
        """
        if prompt is None:
            if self.llava is None:
                self.llava = Llava(device='cpu',llava_ckpt=self.opt.vlm.llava.ckpt)
                self.llava.model.to('cuda:2') # TODO
            prompt = self.llava(image,llava_prompt)
            split  = str.rfind(prompt,'ASSISTANT: This image is taken from a scene of ') + len(f'ASSISTANT: This image is taken from a scene of ')
            prompt = prompt[split:]
            print(prompt) 
            # self.llava.model.to('cpu')
        return prompt
    
    def _inpaint(self, image:Image.Image, mask:Image.Image, prompt=None, negative_prompt='any fisheye, any large circles, any blur, unrealism.', size=None):
        '''mask=1: for inpainting, mask=0: fixed'''
        if size== None:
            size = self.size
            
        self.flux.pipe= self.flux.pipe.to('cuda') # TODO
        result = self.flux.pipe(
            prompt=prompt,
            height=size[1],
            width=size[0],
            control_image=image,
            control_mask=mask,
            num_inference_steps=self.opt.inpaint.flux.num_inference_steps,
            generator=self.generator,
            controlnet_conditioning_scale=self.opt.inpaint.flux.controlnet_conditioning_scale,
            guidance_scale=self.opt.inpaint.flux.guidance_scale,
            negative_prompt=negative_prompt,
            true_guidance_scale=self.opt.inpaint.flux.true_guidance_scale
        ).images[0]
        self.flux.pipe = self.flux.pipe # switch to cpu later?
        return result
    
    def outpaint(self, image, scale=None, prompt=None, negative_prompt='any fisheye, any large circles, any blur, unrealism.', save=True):
        image = convert_PIL(image)
        ori_image, image, mask = expand_and_mask_image(image,scale=scale)
        prompt = self.get_prompt(ori_image, prompt)
        result = self._inpaint(image, mask, prompt, negative_prompt)
        if save:
            result.save(self.opt.inpaint.outpaint_image_path)
        return result
    
    def inpaint_bg(self, image, mask, prompt=None, negative_prompt='',llava_prompt=None, save=True):
        image = convert_PIL(image)
        mask= 1.0 - mask 
        mask = binary_dilation(mask, iterations=20)
        Image.fromarray((mask * 255).astype(np.uint8)).save('cache/dilated_mask.png')
        mask = convert_PIL(mask).convert('L')
        prompt = self.get_prompt(image, prompt=prompt,llava_prompt=llava_prompt)
        result = self._inpaint(image, mask, prompt, negative_prompt)
        if save:
            result.save(self.opt.inpaint.inpaint_bg_path)
        return result
    
    def inpaint_obj(self,image,prompt,negative_prompt=''):
        """
            image: [H, W, 4]
        """
        size = (512,512)
        image, mask =resize_and_center(image,size=size)
        image = convert_PIL(image)
        mask= 1.0 - mask 
        Image.fromarray((mask * 255).astype(np.uint8)).save('cache/obj_mask.png')
        prompt = self.get_prompt(image, prompt=prompt)
        result = self._inpaint(image, mask, prompt, negative_prompt,size=size)
        result.save('cache/inpaint_obj.png')
        return result
        
        
        
    
