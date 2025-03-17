# Given an image and camera trajectory, output a video
from PIL import Image
import numpy as np
import os
from omegaconf import OmegaConf

from ops.cam_utils import CamPlanner, TrajMove
from ops.cam_utils import Mcam
from ops.utils.general import seed_everything
from pipe.view_extend import *
from ops.utils.general import easy_save_video

traj = CamPlanner().add_traj().move_forward(0.1,num_frames=48).finish() # You can define your trajectory here

def dust3r_pipe(opt):
    from ops.dust3r import Dust3rWrapper
    from ops.PcdMgr import PcdMgr
    global dust3r
    dust3r = Dust3rWrapper(opt.dust3r)
    # preprocess input
    dust3r.load_initial_images([input_image_path],opt)
    global H, W
    H, W = dust3r.images[0]["img"].shape[2:]
    
    # background_mask = get_real_bg_mask(bg_mask)
    background_mask=None
    # generate scene
    dust3r.run_dust3r_init(bg_mask=background_mask)
    bg_pm, pm = dust3r.get_inital_pm()
    cam = dust3r.get_cams()[-1]
    pm=pm.cpu().numpy()
    pm[...,:3] = pm[...,:3] @ cam.getW2C()[:3, :3].T + cam.getW2C()[:3, 3].T
    Mcam.set_default_f(cam.f)
    
    pcdtmp = PcdMgr(pts3d=pm.reshape(-1, 6))
    pcdtmp.save_ply(f"{output_dir}/{name}_pcd_ori.ply")
    
    return pcdtmp

def video_generate(image, traj, prompts=None):
    pcd = dust3r_pipe(opt)
    video_tool = Video_Tool(opt, dust3r)
    for cam in traj:
        cam.set_cam(f = Mcam.default_f)
    # pcd.remove_outliers_near()
    video = video_tool.run(traj,image,pcd,prompts=prompts,logger=False)
    return video

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="test", help='Name of the configuration to load')
    parser.add_argument('--basic_opt', type=str, default='configs/basic.yaml', help='Name of the configuration to load')
    parser.add_argument('--output_dir', type=str, default="./results-single-traj", help='Name of the configuration to load')
    parser.add_argument('--input_image_path', type=str, default="./assets/room.png", help='Name of the configuration to load')
    args = parser.parse_args()
    name = args.name
    output_dir = args.output_dir
    input_image_path = args.input_image_path
    
    basic_opt = OmegaConf.load('configs/basic.yaml')
    opt = OmegaConf.load(f'configs/examples/test.yaml')
    opt = OmegaConf.merge(basic_opt, opt) 
    opt.name = name
    
    os.makedirs(output_dir, exist_ok=True) # create results folder
    seed_everything(opt)
    
    
    image = Image.open(input_image_path).resize((1024,576))
    image = np.array(image.convert("RGB"))
    
    video = video_generate(image, traj)
    easy_save_video(video,f"{output_dir}/{name}.mp4")
