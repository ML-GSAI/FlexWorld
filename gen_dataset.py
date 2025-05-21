import os
import numpy as np
import torch
import multiprocessing
import tqdm
import argparse


from ops.cam_utils import CamPlanner
from ops.gs.base import GaussianMgr
from ops.utils.general import *
from data.dl3dv import DL3DV


def get_skip_frame(total_frames):
    target_frames = 60
    randidx = np.random.randint(1, total_frames-target_frames+1)
    frame_idx = range(randidx, randidx+target_frames)
    return list(frame_idx)
    
def get_skip_2frame(total_frames):

    target_frames = 49
    max_try = 100

    for _ in range(max_try):
        randidx= np.random.randint(1, total_frames)
        if randidx + target_frames * 2 > total_frames:
            continue
        break

    frame_idx = range(randidx, randidx+target_frames*2, 2)
    return list(frame_idx)


def get_2skip_frames(total_frames):

    target_frames = 49
    max_try = 100

    for _ in range(max_try):
        randidx, randidx2 = np.random.randint(1, total_frames-target_frames+1, size=2)

        if abs(randidx - randidx2) >= target_frames:
            break

    frame_idx = range(randidx, randidx+target_frames)
    frame_idx2 = range(randidx2, randidx2+target_frames)
    return list(frame_idx), list(frame_idx2)


    
def get_rounded_frame(total_frames):

    target_frames = 49
    randidx= np.random.randint(1, total_frames - target_frames // 2)

    frame_idx = range(randidx, randidx+target_frames//2 + 1)
    return list(frame_idx) + list(frame_idx)[::-1][1:]

def get_random_pcdidx():
    target_frames = 49
    extranum = np.random.randint(1, 5)
    pcdidx = np.random.randint(1, target_frames, extranum)
    return sorted([0] + list(pcdidx))


def run_one(i):

    dl3dv = DL3DV(dataset_path=dataset_path, output_path=output_path, gspath=gs_path)
    check = dl3dv.try_load_json_and_check(i)
    if not check:
        return True
    total_frames = dl3dv.get_vidlen()

    frame_idx1 = get_skip_frame(total_frames)


    dl3dv.new_save_dir()
    dl3dv.setup_gs_pseduo_res(0, frame_idx1)
    dl3dv.save_gt_video()
    dl3dv.save_broken_video_with_singlepcd(0,name="1")  


    return True


dataset_path="./DL3DV/DL3DV-10K/1K" 
output_path="./DL3DV/processed/1K"
gs_path = "./gaussian-splatting/output"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path")
parser.add_argument("--output_path")
parser.add_argument("--gs_path")

args = parser.parse_args()
dataset_path = args.dataset_path
output_path = args.output_path
gs_path = args.gs_path

CamPlanner.suppress_tqdm = True
GaussianMgr.suppress_warning = True

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn', force=True)
    PROCESSORS = 5
    dl3dv = DL3DV(dataset_path=dataset_path, output_path=output_path, gspath=gs_path)
    arg = list(range(dl3dv.datalen()))
    with multiprocessing.Pool(PROCESSORS) as p:
        ress = p.imap_unordered(run_one, arg)
        for res in tqdm.tqdm(ress, total=len(arg)):
            pass

