# 把数据集格式转成cogvideo的dataset格式
import PIL
import torch
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import cv2
import os
from tqdm import tqdm
import argparse

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
        
        self.prompt= '<image>\n USER: Detaily describe the scene this image taken from? \n ASSISTANT: This image is taken from a scene of ' 
        # self.prompt= '<image>\n USER: Describe the main objects with no imagination. \n ASSISTANT: This image is taken from a scene of ' 
        
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
    
    
    def get_prompt(self, image:Image.Image, prompt=None, llava_prompt=None):
        """
            if prompt is None, use llava to generate prompt
        """
        if prompt is None:
            # self.model.to('cuda:1') # TODO
            prompt = self(image,llava_prompt)
            split  = str.rfind(prompt,'ASSISTANT: This image is taken from a scene of ') + len(f'ASSISTANT: This image is taken from a scene of ')
            prompt = prompt[split:]
            # self.model.to('cpu')
        return prompt


def get_first_frame_as_pil(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    ret, frame = cap.read()
    
    cap.release()
    
    if not ret:
        raise ValueError("无法读取视频帧")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image = Image.fromarray(frame_rgb)
    
    return image


def get_vid_prompt(video_path, llava):
    image = get_first_frame_as_pil(video_path)
    prompt = llava.get_prompt(image)
    return prompt


parser = argparse.ArgumentParser()
parser.add_argument("--output_path")
parser.add_argument("--input_path")

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

os.makedirs(output_path, exist_ok=True)
labels_path=os.path.join(output_path,'labels')
videos_path=os.path.join(output_path,'videos')
os.makedirs(labels_path, exist_ok=True)
os.makedirs(videos_path, exist_ok=True)
llava = Llava()

all_paths=sorted(os.listdir(input_path))

for i,p in enumerate(tqdm(all_paths, desc="Processing paths"), 1):
    try:
        path = os.path.join(input_path, p)
        vid_path=os.path.join(path,'real.mp4')
        prompt=get_vid_prompt(vid_path, llava).strip()
        cond_vid_path=os.path.join(path,'1.mp4')
        os.system(f'cp {vid_path} {os.path.join(videos_path,f"{p}-ref.mp4")}')
        os.system(f'cp {cond_vid_path} {os.path.join(videos_path,f"{p}-cond.mp4")}')
        with open(os.path.join(labels_path,f'{i}.txt'),'w') as f:
            f.write(prompt+'\n')
    except Exception as e:
        print(f'Error: {e}, {p}')
        continue