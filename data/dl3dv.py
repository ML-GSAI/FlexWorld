from omegaconf import OmegaConf
import os
import json
import numpy as np
import glob
from PIL import Image
import torch
from dataclasses import dataclass
from tqdm import tqdm
import shutil
import re


from ops.cam_utils import Mcam, CamPlanner
from ops.gs.base import GaussianMgr
from ops.PcdMgr import PcdMgr
from ops.utils.general import *
from ops.utils.depth import visualize_depth, depth2pcd_world_torch, depth2pcd_world



class DL3DV:
    def __init__(self, dataset_path, output_path, gspath="./gaussian-splatting/output"):
        # dataset_path="./datasets/DLD3DV/DL3DV-10K/1K", output_path="./datasets/processed/gs",
        self.dataset_path = dataset_path
        subset_match = re.search(r'/(\d+K)$', dataset_path)
        self.subsetname = subset_match.group(1) if subset_match else " "
        self.dataset_ext = "images_4/frame_{:05d}.png"

        self.gspath = gspath
        self.gspath_ext = "point_cloud/iteration_30000/point_cloud.ply"
        self.gspath_ext2 = "point_cloud/iteration_7000/point_cloud.ply"

        self.output_path = output_path
        self.folder_names_raw = os.listdir(self.dataset_path)
        self.folder_names = [x for x in self.folder_names_raw if len(x) > 20 and os.path.isdir(os.path.join(self.dataset_path, x))]
        self.folder_names.sort()
        filter_out = set(self.folder_names_raw) - set(self.folder_names)
        #print(f"filtered out {len(filter_out)} filenames:", filter_out)

    def clone(self):
        return DL3DV(self.dataset_path, self.output_path, self.gspath)
    
    def datalen(self):
        return len(self.folder_names)

    def load_json(self, idx):
        '''
        应该首先调用这个函数，然后再调用其他函数
        '''
        self.idx = idx
        self.folder_name = self.folder_names[idx]
        self.gs = None
        self.suffix = None

        path_to_json = os.path.join(self.dataset_path, self.folder_names[idx], "transforms.json")
        if not os.path.exists(path_to_json):
            raise ValueError(f"json not exists")
        with open(path_to_json, "r") as f:
            data = json.load(f)
        self.data = data


    def try_load_json_and_check(self, idx):
        try:
            self.load_json(idx)
            self.get_3dgs()
            cams = self.get_cam(list(range(1,2)))
            vid = self.get_vid(list(range(1,2)))
            if vid[0].shape != (cams[0].H, cams[0].W, 3):
                raise ValueError(f"source image resolution not match {vid[0].shape}")

        except Exception as e:
            error_msg = str(e)
            error_msg = f"{self.subsetname},{idx},{self.folder_name[:10]},{error_msg}"
            filename = "cache/datasets/error.txt"
            with open(filename, "a+") as f:
                f.write(error_msg + "\n")
            print(f"error: {error_msg}")
            return False
        return True


    def _get_idx_from_name(self, name):
        return self.folder_names.index(name)
    
    def _get_name_from_idx(self, idx):
        return self.folder_names[idx]

    def get_3dgs(self, fresh=True):
        if self.gs is None:
            path = os.path.join(self.gspath, self.folder_name, self.gspath_ext)
            if not os.path.exists(path):
                path = os.path.join(self.gspath, self.folder_name, self.gspath_ext2)
            if not os.path.exists(path):
                raise ValueError(f"3dgs not exist")
            self.gs = GaussianMgr().load_ply(path, flip=False)
        return self.gs
    
    def get_vidlen(self):
        pathdir = os.path.join(self.dataset_path, self.folder_name, "images_4")
        allvid = glob.glob(os.path.join(pathdir, "*.png"))
        return len(allvid)
    

    def get_vid(self, framelist=None):
        '''
        idx: the idx of the scene, like 0~999 ?
        framelist: the idx of frames to get, if None, get all frames

        ret: a list of np img, each frame is a numpy array of shape (H, W, 3), range in [0,1]
        '''
        
        if framelist is None:
            l = self.get_vidlen()
            framelist = range(1, l+1)
        imgs = []
        for i in framelist:
            imgpath = os.path.join(self.dataset_path, self.folder_name, self.dataset_ext.format(i))
            img = Image.open(imgpath)
            img = np.array(img) / 255.0
            imgs.append(img)

        return imgs
    

    def get_cam(self, framelist=None):
        # give out OpneCV cams, https://github.com/DL3DV-10K/Dataset/issues/4
        # this maybe uncorrect, because transformer.json may skip frames
        if self.get_vidlen() != len(self.data["frames"]):
            raise ValueError(f"The length of frames in directory ({self.get_vidlen()}) is not equal to the length of transformer.json ({len(self.data['frames'])})")
        data = self.data
        RTs = [np.array(x["transform_matrix"]) for x in data["frames"]]
        c2ws = []
        for RT in RTs:
            c2w = RT
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
            c2ws.append(c2w.copy())

        # test real resolution
        real_h = 540
        ratio = data["h"] / real_h

        # set cam
        cam = Mcam()
        cam.H = int(data["h"] / ratio)
        cam.W = int(data["w"] // ratio)
        cam.f = data["fl_x"] / ratio # we only use fx here
        cam.set_default_c()

        cams = [cam.copy() for _ in range(len(c2ws))]
        for i, cam in enumerate(cams):
            cam.R = c2ws[i][:3, :3]
            cam.R[:, [1,2]] = -cam.R[:, [1,2]]
            cam.T = c2ws[i][:3, 3]
        
        if framelist is not None:
            cams = [cams[i-1] for i in framelist]
        return cams
    
    def get_skip_frame(self):
        '''
        在load_json之后调用
        挑帧逻辑在此，后期需要优化, 其中元素应该落在1~len(data["frames"])之间
        '''
        total_frames = self.get_vidlen()

        frame_idx = range(1, 51)
        return list(frame_idx)
    
    def setup_result(self, pms, cams, frameidx):
        name = self.folder_name
        res = Mast3rResult(pms=pms, cams=cams, frameidx=frameidx, name=name)
        res.check()
        self.res = res
        return res
    
    def set_save_suffix(self, suffix):
        self.suffix = suffix

    def new_save_dir(self):
        '''自动分配一个新的保存路径,通过一个新的suffix'''
        if self.suffix is None:
            self.suffix = 1
        else:
            self.suffix += 1


    def get_save_dir(self):
        if self.suffix is not None:
            dirpath = os.path.join(self.output_path, self.res.name + "_" + str(self.suffix))
        else:
            dirpath = os.path.join(self.output_path, self.res.name)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        return dirpath


    def save(self, ptname = "result.pkl"):
        if self.res is None:
            raise ValueError("You should call setup_result first")
        res = self.res
        dirpath = self.get_save_dir()
        path = os.path.join(dirpath, ptname)
        torch.save(res, path)

    def load(self, name=None, ptname = "result.pkl"):
        '''
        加载发生时,我们不需要继续工作了,因此重设result,冷启动需要调用load_json
        '''
        if name is None:
            name = self.folder_name
            if name is None:
                raise ValueError("You should provide a name or load_json first")
        path = os.path.join(self.output_path, name, ptname)
        idx = self._get_idx_from_name(name)
        res = torch.load(path)
        self.res = res
        return res
    
    # ----------------- 以下函数为高斯服务 -----------------
    def setup_gs_pseduo_res(self, pcdidx, frameidx=None):
        '''
        根据当前idx生成一个假的res,深度来自渲染高斯,相机姿态即为数据集姿态
        注意:只有self.res.pm[pcdidx]是有意义的!
        如果需要都有意义,调用setup_gs_pseudo_res_all
        pcdidx: 0 ~ len(frameidx)-1
        '''
        if frameidx is None:
            frameidx = self.get_skip_frame()
        cams = self.get_cam(frameidx)
        cols = self.get_vid(frameidx)
        gs = self.get_3dgs()


        cam = cams[pcdidx]
        depth = gs.render(cams[pcdidx])[1]
        pts = depth2pcd_world_torch(depth, cam)
        col = torch.tensor(cols[pcdidx], device=pts.device)
        pts6d = torch.concatenate([pts, col], axis=-1).reshape(-1, 6)
        pms = [pts6d for _ in frameidx]
        
        self.setup_result(pms, cams, frameidx)

    def setup_gs_pseudo_res_some(self, pcdidx_list, frameidx=None):
        '''pcdidx_list: 内部的元素应该在0-len(frameidx)-1之间'''
        self.get_3dgs(fresh=True)
        if frameidx is None:
            frameidx = self.get_skip_frame()
        pms = [None for _ in frameidx]
        for idx in pcdidx_list:
            self.setup_gs_pseduo_res(idx, frameidx=frameidx)
            pms[idx] = self.res.pms[idx]
        cams = self.res.cams
        self.setup_result(pms, cams, frameidx)
    

    def setup_gs_pseudo_res_all(self, frameidx=None):
        self.get_3dgs(fresh=True)
        if frameidx is None:
            frameidx = self.get_skip_frame()
        pms = []
        for idx in range(len(frameidx)):
            self.setup_gs_pseduo_res(idx, frameidx=frameidx, fresh=False)
            pms.append(self.res.pms[idx])
        cams = self.res.cams
        self.setup_result(pms, cams, frameidx)

    # ----------------- 以下函数需要self.res -----------------
    def save_gt_video(self, name="real"):
        vid = self.get_vid(self.res.frameidx)
        path = os.path.join(self.get_save_dir(), f"{name}.mp4")
        easy_save_video(vid, path, fps=8, value_range="0,1")

    def save_broken_video_with_singlepcd(self, pcdidx, name="1"):
        self.save_broken_video_with_somepcd([pcdidx], name=name)

    def save_broken_video_with_allpcd(self, name="all"):
        self.save_broken_video_with_somepcd(list(range(len(self.res.pms))), name=name)

    def save_broken_video_with_somepcd(self, pcdidx, name="f20"):
        ''' pcdidx: list of int e.g. [0,1,2]
        '''
        pms = [self.res.pms[i] for i in pcdidx]
        pcd = PcdMgr(pts3d=pms)
        path = os.path.join(self.get_save_dir(), f"{name}.mp4")
        cams = self.res.cams

        backend = PcdMgr.get_default_render_backend()
        PcdMgr.set_default_render_backend("gs")
        CamPlanner._render_video(pcd, cams, output_path=path, fps=8)
        PcdMgr.set_default_render_backend(backend)

        


@dataclass
class Mast3rResult():
    pms: list # F [N, 6] np.ndarray
    frameidx: list # F int
    cams: list # F Mcams
    name: str # scene id

    def check(self):
        assert isinstance(self.pms, list), type(self.pms)
        assert isinstance(self.frameidx, list), type(self.frameidx)
        assert isinstance(self.cams, list), type(self.cams)
        
        assert isinstance(self.frameidx[0], int), type(self.frameidx[0])
        assert isinstance(self.cams[0], Mcam), type(self.cams[0])

        assert len(self.pms) == len(self.frameidx), (len(self.pms), len(self.frameidx))
        assert len(self.frameidx) == len(self.cams), (len(self.frameidx), len(self.cams))
        assert len(self.pms[0].shape) == 2, len(self.pms[0].shape)
        assert self.pms[0].shape[1] == 6, self.pms.shape[1]


# load_json -> get_skip_frame -> get_vid -> run_dust3r -> save -> 
