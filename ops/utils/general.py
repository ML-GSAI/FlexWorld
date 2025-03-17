import imageio
import numpy as np
import torch
from PIL import Image
from typing import Literal
import os
import random
import torchvision
import einops

def convert_PIL(image):
    """
        image can be a PIL Image, a path to an image, a numpy array ([H, W, c]) or a torch tensor ([H, W, c]).
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        return Image.open(image)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            return Image.fromarray(image)
        else:
            image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image)
    else:
        raise ValueError("Unsupported type of image")


def to_numpy(image):
    """
        image can be a PIL Image, a path to an image, a numpy array ([H, W, c]) or a torch tensor ([H, W, c]).
    """
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, str):
        return np.array(Image.open(image))
    elif isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    else:
        raise ValueError("Unsupported type of image")
    
def to_HWC(frames):
    '''
    Turn ...CHW or ...HWC to ...CHW. C = 1 or 3.
    '''
    if frames.ndim == 2:
        frames = frames[None, ...]

    if frames.ndim < 3 or (frames.shape[-3] not in [1, 3] and frames.shape[-1] not in [1, 3]):
        raise ValueError("The input frames must have 1 or 3 channels, got {}.".format(frames.shape))

    if frames.shape[-3] in [1, 3]:
        frames = einops.rearrange(frames, '... c h w -> ... h w c')
    return frames



def infer_value_0_255(value, value_range: Literal["0,1", "0,255", "-1,1"] = None):
    '''
    Convert the input to [0, 255] with uint8.
    Parameters:
        value (numpy.ndarray): Input value.
    Returns:
        numpy.ndarray: The converted value in the range of [0, 255] with np.uint8 dtype.
    '''

    if value_range is None:
        if value.max() > 10:
            value_range = "0,255"
        elif value.min() < -0.1:
            value_range = "-1,1"
        else:
            value_range = "0,1"
        print(f"infer value_range: {value_range}")
        
        
    if value_range == "0,255":
        value = value.astype(np.uint8)
    elif value_range == "-1,1":
        value = ((value + 1) * 127.5).astype(np.uint8)
    elif value_range == "0,1":
        value = (value * 255).astype(np.uint8)

    return value
    

def easy_save_video(frames, output_path, fps=8, value_range: Literal["0,1", "0,255", "-1,1"] = None):
    """
    Save video using imageio with automatic handling of value ranges and input formats.

    Parameters:
        frames (numpy.ndarray, torch.Tensor, or list): Input video frames with shape (F, C, H, W) or (F, H, W, C). C = 1 or 3.
        output_path (str): The output file name (e.g., 'output.mp4').
        fps (int): Frames per second for the output video. Default is 30.
        value_range (str): The value range of the input frames. Default is None.
            - '0,1': The input frames are in the range of [0, 1].
            - '0,255': The input frames are in the range of [0, 255].
            - '-1,1': The input frames are in the range of [-1, 1].
    """

    if isinstance(frames, torch.Tensor):
        frames = to_numpy(frames)
    elif isinstance(frames, list):
        if len(frames) == 0:
            raise ValueError("The input list is empty.")
        # first convert list elements to numpy arrays
        elem = frames[0]
        if isinstance(elem, torch.Tensor):
            frames = [to_numpy(img) for img in frames] 
        elif isinstance(elem, Image.Image):
            frames = [np.array(img) for img in frames] 
        
        # then convert list to numpy array
        frames = np.stack(frames, axis=0)
    
    if frames.ndim != 4:
        raise ValueError("Input frames must be a 4D array with shape (F, C, H, W) or (F, H, W, C).")
        
    # now we have a numpy array with shape (F, C, H, W) or (F, H, W, C), convert to (F, H, W, C)
    frames = to_HWC(frames)

    frames = infer_value_0_255(frames, value_range)

    imageio.mimsave(output_path, frames, fps=fps, macro_block_size = 4)

def save_video(data,images_path,folder=None):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder]*len(data)
        images = [np.array(Image.open(os.path.join(folder_name,path))) for folder_name,path in zip(folder,data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=8, video_codec='h264', options={'crf': '10'})


def seed_everything(opt):
    try:
        seed = int(opt.seed)
    except:
        seed = np.random.randint(0, 1000000)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def extract_video_to_images(src, dst = None):
    video_dir = os.path.dirname(src)
    video_name = os.path.splitext(os.path.basename(src))[0]
    if dst is None:    
        # 创建保存图片的目录
        dst = os.path.join(video_dir, f"{video_name}_frames")
    
    os.makedirs(dst,exist_ok=True)
    vid = imageio.mimread(src)
    for i,img in enumerate(vid):
        Image.fromarray(img).save(os.path.join(dst,f"{video_name}_frames{i:04d}.png"))

def extract_all_videos_in_folder(folder):
    import glob
    for video in glob.glob(os.path.join(folder,"**/*.mp4"),recursive=True):
        print(video)
        extract_video_to_images(os.path.join(folder,video),os.path.join(folder,video.split(".")[0]))