import PIL
import torch
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration

class Llava():
    def __init__(self,device='cuda',
                 llava_ckpt='llava-hf/bakLlava-v1-hf') -> None:
        self.device = device
        self.model_id = llava_ckpt
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # self.prompt= '<image>\n USER: Detaily imagine and describe the scene this image taken from? \n ASSISTANT: This image is taken from a scene of ' 
        self.prompt= '<image>\n USER: Describe the main objects with no imagination. \n ASSISTANT: This image is taken from a scene of ' 
        
    def __call__(self,image:PIL.Image, prompt=None):

        # input check
        if not isinstance(image,PIL.Image.Image):
            if np.amax(image) < 1.1:
                image = image * 255
            image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)
        
        prompt = self.prompt if prompt is None else prompt
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device,torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return answer