from ops.gs.base import Trainable_Gaussian, GaussianMgr
from ops.gs.train import GS_Train_Tool, GS_Train_Config
import torch
import torch.nn.functional as F

class GS_Train_Pipe():
    def __init__(self, opt, pts, device='cuda'):
        self.opt = opt
        self.gs = GaussianMgr().init_from_pts(pts, mode="fixed", scale = 3e-4)
        self.gs_cfg = GS_Train_Config()
        try:
            for attr in dir(opt.gs):
                if hasattr(self.gs_cfg, attr):
                    setattr(self.gs_cfg, attr, getattr(opt.gs, attr))
        except:
            pass
        self.trainer = GS_Train_Tool(self.gs,self.gs_cfg)
        
        if opt.sr:
            from ops.sr_tool import SRTool
            self.sr_tool = SRTool()
        else:
            self.sr_tool = None
        
        self.ref_img = None
        self.ref_cam = None
        self.current_trajs = []
        self.videos = []
        self.masks = []
        self.device = device
        
    def add_ref(self, weight=30):
        if self.ref_img is not None:
            for i in range(weight):
                self.trainer.add_trainable_frame(self.ref_img, self.ref_cam)
    
    def add_frame(self, ref_vid, cam_trajs, masks=None):
        if self.sr_tool is not None:
            ref_vid = self.sr_tool(ref_vid, upscale=2.0).float()
        for i in range(len(cam_trajs)):
            cam_trajs[i].set_size(ref_vid[i].shape[0], ref_vid[i].shape[1])
        if self.ref_img is None:
            self.ref_img = ref_vid[0]
            self.ref_cam = cam_trajs[0]
        self.current_trajs.append(cam_trajs)
        self.videos.append(ref_vid)
        if masks is not None:
            for i in range(len(masks)):
                masks[i] = F.interpolate(torch.tensor(masks[i], dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0), size=ref_vid[i].shape[0:2], mode='bilinear', align_corners=False).squeeze(0).squeeze(0).bool()
            self.masks.append(masks)
            
    def train(self,iters=3000, enable_cam = True, full_train = True, add_ref=30):
        """
            full_train: if True, train all videos, else only train the last video
        """
        if full_train:
            self.trainer.add_trainable_videos(self.videos, self.current_trajs)
        else:
            self.trainer.add_trainable_videos([self.videos[-1]], [self.current_trajs[-1]])
        self.add_ref(weight=add_ref)
        gs_train = self.trainer.train(iters=iters, enable_densification=True,enable_cam=enable_cam)
        self.gs = gs_train.getMgr()
        self.trainer = GS_Train_Tool(self.gs,self.gs_cfg)
        return self.gs
    
    def train_with_mask(self,iters=1000, enable_cam = True, full_train = True):
        """
            full_train: if True, train all videos, else only train the last video
        """
        if full_train:
            self.trainer.add_trainable_videos_and_mask(self.videos, self.current_trajs,self.masks)
        else:
            self.trainer.add_trainable_videos_and_mask([self.videos[-1]], [self.current_trajs[-1]],[self.masks[-1]])
        gs_train = self.trainer.train_with_mask(iters=iters, enable_densification=True)
        self.gs = gs_train.getMgr()
        self.trainer = GS_Train_Tool(self.gs,self.gs_cfg)
        return self.gs
    
    def gs_update(self, gs):
        self.gs = gs
        self.trainer = GS_Train_Tool(self.gs,self.gs_cfg)
        return self.gs