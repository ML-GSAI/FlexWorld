from ops.cam_utils import Mcam,CamPlanner
from ops.cam_utils import CamPlanner
from ops.PcdMgr import PcdMgr
from ops.utils.general import save_video
from ops.gs.base import GaussianMgr
from ops.dust3r import Dust3rWrapper
from ops.gs.train import GS_Train_Tool, GS_Train_Config

import numpy as np
import torch
from PIL import Image
import imageio
import os
from omegaconf import OmegaConf
from ops.dust3r import Dust3rWrapper
from ops.utils.general import save_video,extract_video_to_images,easy_save_video
from ops.cam_utils import Mcam
import tqdm
import re




valid_move_instructs = ["up","down","left","right","forward","backward","rotate_left","rotate_right"]

def get_traj_simple(move_instruct):
    if move_instruct not in valid_move_instructs:
        raise ValueError("Invalid move_instruct: {}, should be in {}".format(move_instruct, valid_move_instructs))
    if move_instruct == "up":
        traj = CamPlanner().add_traj().move_up(0.1,num_frames=48).finish()
    elif move_instruct == "down":
        traj = CamPlanner().add_traj().move_up(-0.1,num_frames=48).finish()
    elif move_instruct == "left":
        traj = CamPlanner().add_traj().move_left(0.1,num_frames=48).finish()
    elif move_instruct == "right":
        traj = CamPlanner().add_traj().move_left(-0.1,num_frames=48).finish()
    elif move_instruct == "forward":
        traj = CamPlanner().add_traj().move_forward(0.1,num_frames=48).finish()
    elif move_instruct == "backward":
        traj = CamPlanner().add_traj().move_forward(-0.1,num_frames=48).finish()
    elif move_instruct == "rotate_left":
        traj = CamPlanner().add_traj().rotate(20,num_frames=48).finish()
    elif move_instruct == "rotate_right":
        traj = CamPlanner().add_traj().rotate(-20,num_frames=48).finish()
    return traj




def get_traj_video(vidpath):
    basic_opt = OmegaConf.load('configs/basic.yaml')
    opt = OmegaConf.load(f'configs/examples/house.yaml')
    opt = OmegaConf.merge(basic_opt, opt) 

    dust3r = Dust3rWrapper(opt.dust3r,load_mast3r=True)
    images = imageio.mimread(vidpath)
    images = np.stack(images) / 255.0

    images_mast3r = dust3r._load_our_images(images)
    global H, W
    H, W = images_mast3r[0]["img"].shape[2:]
    
    scene = dust3r.run_mast3r(images_mast3r)
    cams = dust3r.get_cams_mast3r(scene)

    
    return cams

    
def get_traj(traj_ins):
    if traj_ins in valid_move_instructs:
        traj = get_traj_simple(traj_ins)
    elif os.path.isfile(traj_ins):
        traj = get_traj_video(traj_ins)
    else:
        raise ValueError("Invalid traj: {}, should be in {} or a valid video file".format(traj_ins, valid_move_instructs))
    return traj