from dataclasses import dataclass
import tqdm
import torch
# import lpips
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as tvtf
from torchmetrics.image import StructuralSimilarityIndexMeasure
import PIL.Image
import gsplat

from ops.gs.base import Trainable_Gaussian, GaussianMgr
from ops.cam_utils import Mcam
from ops.gs.deformation import deform_network, SimpleDeform
from ops.utils.logger import Ulog


class RGB_Loss():
    def __init__(self,w_l1=1.0,w_lpips=0.0,w_ssim=0.2):
        self.rgb_loss = F.smooth_l1_loss
        if w_lpips > 0:
            import lpips
            self.lpips_alex = lpips.LPIPS(net='alex').to('cuda')
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        
    def __call__(self,pr,gt,valid_mask=None):
        pr = torch.nan_to_num(pr)
        gt = torch.nan_to_num(gt)
        if len(pr.shape) < 3: pr = pr[:,:,None].repeat(1,1,3)
        if len(gt.shape) < 3: gt = gt[:,:,None].repeat(1,1,3)
        pr_valid = pr[valid_mask] if valid_mask is not None else pr.reshape(-1,pr.shape[-1])
        gt_valid = gt[valid_mask] if valid_mask is not None else gt.reshape(-1,gt.shape[-1])
        l_rgb = self.rgb_loss(pr_valid,gt_valid)
        l_ssim = 1.0 - self.ssim(pr[None].permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2))
        l_lpips = 0.0
        if self.w_lpips > 0:
            l_lpips = self.lpips_alex(pr[None].permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2))
        return self.w_l1 * l_rgb + self.w_ssim * l_ssim + self.w_lpips * l_lpips
    


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.n = n

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = CameraOptModule.rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)
    
    def get_all_transform(self):
        embed_ids = torch.arange(self.n, device=self.embeds.weight.device)
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = CameraOptModule.rotation_6d_to_matrix(
            drot + self.identity.expand(self.n, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((self.n, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return transform

    
    @staticmethod
    def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)


@dataclass
class GS_Train_Config():

    # hyperparameters for strategy (prune, densify, and update)
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 0    # be modified
    refine_stop_iter: int = 15_000
    reset_every: int = 300000 # we do not want reset
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = True


    # loss params
    w_rgb:float = 1.0
    w_ssim:float = 0.2
    w_lpips:float = 0.0
    # learning rate
    xyz_lr:float = 0.00001
    rgb_lr:float = 0.0005
    opacity_lr:float = 0.005
    scale_lr:float = 0.0005
    rotation_lr:float = 0.0001

    # param for camera optimization
    enable_cam:bool = False
    cam_lr:float = 1e-5
    cam_rg:float = 1e-6

    # param for deformation 
    enable_deform:bool = False
    deform_mlp_lr:float = 0.00016
    deform_grid_lr:float = 0.0016

    # iters for training
    iters:int = 100

class GS_Train_Tool():
    '''
    Frames and well-trained gaussians are kept, refine the trainable gaussians
    The supervision comes from the Frames of GS_Scene
    '''
    def __init__(self, GS:Trainable_Gaussian|GaussianMgr, opt:GS_Train_Config = GS_Train_Config()):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opt = opt
        if isinstance(GS, GaussianMgr):
            GS = Trainable_Gaussian(GS)
        self.GS = GS

        self.rgb_lossfunc = RGB_Loss(w_lpips=opt.w_lpips, w_ssim=opt.w_ssim, w_l1=opt.w_rgb)
        self.refimgs = []
        self.refcams = []
        self.refmask = []
        self.refvideo_list = []
        self.reftraj_list = []
        
        self.strategy = gsplat.DefaultStrategy()
        for key, value in opt.__dict__.items():
            if hasattr(self.strategy, key):
                setattr(self.strategy, key, value)

        self.disable_densification()

    def _init_optimizer_dict(self, param_dict):
        lr_map = {
            "means": self.opt.xyz_lr,
            "colors": self.opt.rgb_lr,
            "scales": self.opt.scale_lr,
            "opacities": self.opt.opacity_lr,
            "quats": self.opt.rotation_lr
        }
        #param_dict = self.GS.getgsplat_dict()
        self.optimizer_dict = {}
        for key_name in param_dict.keys():
            #params = {'params': param_dict[key_name], 'lr': lr_map[key_name], 'name': key_name}
            self.optimizer_dict[key_name] = torch.optim.Adam([param_dict[key_name]], lr=lr_map[key_name])


    def add_trainable_videos(self, videos, trajs_list):
        assert isinstance(videos, list) and isinstance(trajs_list, list), "videos and trajs_list should be list"
        for video, trajs in zip(videos, trajs_list):
            self.add_trainable_video(video, trajs)
            
    def add_trainable_videos_and_mask(self, videos, trajs_list, masks_list):
        assert isinstance(videos, list) and isinstance(trajs_list, list), "videos and trajs_list should be list"
        for video, trajs, masks in zip(videos, trajs_list, masks_list):
            self.add_trainable_video_and_mask(video, trajs, masks)

    def add_trainable_video(self, video, traj):
        if isinstance(video, list):
            N = len(video)
        else:
            N = video.shape[0]
        for i in range(N):
            if traj[i].H != video[i].shape[0] or traj[i].W != video[i].shape[1]:
                print("Warning: not matching size for video and traj, resizing traj to match video")
                break
        for i in range(N):
            self.add_trainable_frame(video[i], traj[i].copy().set_size(video[i].shape[0], video[i].shape[1]))
        self.refvideo_list.append(video)
        self.reftraj_list.append(traj)
        
    def add_trainable_video_and_mask(self, video, traj, mask):
        if isinstance(video, list):
            N = len(video)
        else:
            N = video.shape[0]
        for i in range(N):
            if traj[i].H != video[i].shape[0] or traj[i].W != video[i].shape[1]:
                print("Warning: not matching size for video and traj, resizing traj to match video")
                break
        for i in range(N):
            self.add_trainable_frame_and_mask(video[i], traj[i].copy().set_size(video[i].shape[0], video[i].shape[1]),mask[i])
        self.refvideo_list.append(video)
        self.reftraj_list.append(traj)

    def raise_probability(self, frame_idx:list, multiplier=2):
        '''
        raise the probability of the frame to be selected
        '''
        low = 1 / (len(frame_idx) * (multiplier - 1) + len(self.refimgs))
        high = low * multiplier
        self.probability = np.ones(len(self.refimgs)) * low
        self.probability[frame_idx] = high
        self.manual_prob = True
        return self.probability
    
    def default_probability(self):
        '''
        set up default prob property, basically self.probability and self.manual_prob
        '''
        if hasattr(self, "manual_prob") and self.manual_prob:
            self.manual_prob = False
            return
        low = 1 / len(self.refimgs)
        self.probability = np.ones(len(self.refimgs)) * low
        return self.probability


    def add_trainable_frame(self, image, cam:Mcam):
        '''
        image: [H,W,3] tensor|ndarray|PIL.Image, value range should be 0-1
        '''
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image).astype(np.float32)/255.0)
            print("image is PIL.Image.Image")

        H, W, C = image.shape
        assert C == 3, f"Only HW3 image is supported now, got img shape {image.shape}"
        assert H == cam.H and W == cam.W, f"Image size should match camera size, \
                                            got H={H},W={W} and cam.H={cam.H},cam.W={cam.W}"
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device, dtype=torch.float32)
        elif isinstance(image, torch.Tensor):
            image = image.to(self.device, dtype=torch.float32)
        else:
            raise TypeError("Only support np.ndarray and torch.Tensor for image")
        if image.max() > 10:
            raise ValueError("Image value range should be 0-1, got mean value {}".format(image.mean()))
            
        self.refimgs.append(image)
        self.refcams.append(cam)

    def add_trainable_frame_and_mask(self, image, cam:Mcam, mask):
        '''
        image: [H,W,3] tensor|ndarray|PIL.Image, value range should be 0-1
        mask: [H,W] 0:fixed 1:training signal
        '''
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image).astype(np.float32)/255.0)
            print("image is PIL.Image.Image")

        H, W, C = image.shape
        assert C == 3, f"Only HW3 image is supported now, got img shape {image.shape}"
        assert H == cam.H and W == cam.W, f"Image size should match camera size, \
                                            got H={H},W={W} and cam.H={cam.H},cam.W={cam.W}"
        assert image.shape[:2] == mask.shape, f"Image and mask shape should match, \
                                            got image shape {image.shape} and mask shape {mask.shape}"
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
        elif isinstance(image, torch.Tensor):
            image = image.to(self.device)
        else:
            raise TypeError("Only support np.ndarray and torch.Tensor for image")
        if image.max() > 10:
            raise ValueError("Image value range should be 0-1, got mean value {}".format(image.mean()))
            
        self.refimgs.append(image)
        self.refcams.append(cam)
        self.refmask.append(mask)

    def _render_w2c(self, cam:Mcam, w2c):
        '''
        render with cam, but overwrite the w2c with the given w2c to retain the grad
        w2c: [B=1, 4,4] tensor
        '''
        device = 'cuda' 
        background_color = (0,0,0)

        background = torch.empty((1,3), dtype=torch.float32, device=device)
        background[:,0:3] = torch.tensor(background_color,device=device)

        param = self.GS.get_packed_tensor().to(device)
        xyz, rgb, opacity, scale, rotation = GaussianMgr._unpack_data(param)

        rgb       = GaussianMgr.rgb_act(rgb)
        scale     = GaussianMgr.scale_act(scale)
        rotation  = F.normalize(rotation,dim=1)
        opacity   = GaussianMgr.opacity_act(opacity)

        # gsplat need [M,] for opacity
        opacity   = opacity.squeeze(-1)

        H, W = cam.H, cam.W

        intrinsic = torch.from_numpy(cam.getK()).to(device)
        extrinsic = w2c
        # OpenGL to OpenCV
        extrinsic[:, [1,2],:] = -extrinsic[:, [1,2],:]
        render_out,render_alpha,info = gsplat.rendering.rasterization(means = xyz,
                                                scales    = scale,
                                                quats     = rotation,
                                                opacities = opacity,
                                                colors    = rgb,
                                                Ks        = intrinsic[None],
                                                viewmats  = extrinsic,
                                                width     = W, 
                                                height    = H, 
                                                packed    = False,
                                                near_plane= 0.01,
                                                render_mode="RGB+ED",
                                                backgrounds=background) # render: 1*H*W*(3+1)
        render_out  = render_out.squeeze() # result: H*W*(3+1)
        render_rgb  = render_out[:,:,0:3]
        render_dpt  = render_out[:,:,-1]
        return render_rgb, render_dpt, render_alpha, info
    
    def _render_w2c_deform(self, cam:Mcam, w2c, vid_idx):
        '''
        render with cam, but overwrite the w2c with the given w2c to retain the grad
        w2c: [B=1, 4,4] tensor
        '''
        device = 'cuda' 
        background_color = (0,0,0)

        background = torch.empty((1,3), dtype=torch.float32, device=device)
        background[:,0:3] = torch.tensor(background_color,device=device)

        param = self.GS.get_packed_tensor().to(device)
        xyz, rgb, opacity, scale, rotation = GaussianMgr._unpack_data(param)

        rgb       = GaussianMgr.rgb_act(rgb)
        scale     = GaussianMgr.scale_act(scale)
        rotation  = F.normalize(rotation,dim=1)
        opacity   = GaussianMgr.opacity_act(opacity)
                    
        time=torch.tensor(vid_idx).to(xyz.device)

        xyz, scale, rotation, opacity, rgb = self.deform_net(xyz, scale, rotation, opacity, rgb, time)

        # gsplat need [M,] for opacity
        opacity   = opacity.squeeze(-1)

        H, W = cam.H, cam.W

        intrinsic = torch.from_numpy(cam.getK()).to(device)
        extrinsic = w2c
        # OpenGL to OpenCV
        extrinsic[:, [1,2],:] = -extrinsic[:, [1,2],:]
        render_out,render_alpha,info = gsplat.rendering.rasterization(means = xyz,
                                                scales    = scale,
                                                quats     = rotation,
                                                opacities = opacity,
                                                colors    = rgb,
                                                Ks        = intrinsic[None],
                                                viewmats  = extrinsic,
                                                width     = W, 
                                                height    = H, 
                                                packed    = False,
                                                near_plane= 0.01,
                                                render_mode="RGB+ED",
                                                backgrounds=background) # render: 1*H*W*(3+1)
        render_out  = render_out.squeeze() # result: H*W*(3+1)
        render_rgb  = render_out[:,:,0:3]
        render_dpt  = render_out[:,:,-1]
        return render_rgb, render_dpt, render_alpha, info

    def _render(self, cam:Mcam):
        rgb,dpt,alpha,info = self.GS.render(cam)
        return rgb,dpt,alpha,info
    
    def disable_densification(self):
        self.strategy.refine_stop_iter = -1

    def enable_densification(self, num=None):
        if num is None:
            num = self.opt.iters + 1
        self.strategy.refine_stop_iter = num

    def train(self, iters=None, confidence=None, enable_densification=False, enable_cam=None):
        ''' confidence: [N,H,W] tensor, same as refimgs, confidence map for the loss function '''
        refimg_shape = self.refimgs[0].shape
        N = len(self.refimgs)
        confidence = torch.ones((N,)+refimg_shape) if confidence is None else confidence
        assert confidence.shape[0] == N and confidence.shape[1:] == refimg_shape,\
              f"Confidence shape {confidence.shape} should match refimgs shape {N},{refimg_shape}"
        iters = self.opt.iters if iters is None else iters
        if enable_densification:
            self.enable_densification(iters)
        pbar = tqdm.tqdm(range(1, iters+1), desc="Training", leave=True)
        strategy = self.strategy
        print(strategy)

        if enable_cam is None:
            enable_cam = self.opt.enable_cam
        if enable_cam:
            self.pose_adjust = CameraOptModule(len(self.refimgs)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=self.opt.cam_lr,
                    weight_decay=self.opt.cam_rg,
                )
            ]

        param_dict = torch.nn.ParameterDict(self.GS.getgsplat_dict())
        self._init_optimizer_dict(param_dict=param_dict)
        strategy.check_sanity(param_dict, self.optimizer_dict)
        state = strategy.initialize_state()
        self.default_probability()
        for iter in pbar:
            frame_idx = np.random.choice(len(self.refimgs), 1, p=self.probability)[0]
            frame = self.refimgs[frame_idx]
            cam = self.refcams[frame_idx]
            
            w2c = cam.getW2C()
            if enable_cam:
                w2c_adjusted = self.pose_adjust(torch.tensor(w2c[None]).to(self.device), torch.tensor([frame_idx]).to(self.device))
                render_rgb, _, _, info = self._render_w2c(cam, w2c_adjusted)
            else:
                render_rgb, _, _, info = self._render(cam)
            loss_rgb = self.rgb_lossfunc(render_rgb, frame)
            strategy.step_pre_backward(param_dict, self.optimizer_dict, state, iter, info)
            loss = loss_rgb
            pbar.set_postfix({"Loss": loss.item()})
            loss.backward()  
            if iter % strategy.reset_every == 0:
                print("Reset will happen !!!")
            strategy.step_post_backward(param_dict, self.optimizer_dict, state, iter, info)
            self.GS.setgsplat_dict(param_dict)
            for optimizer in self.optimizer_dict.values():
                optimizer.step()
                optimizer.zero_grad()
            if enable_cam:
                for optimizer in self.pose_optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
        refined_scene = self.GS
        return refined_scene
    

    def check_deform_sanity_and_mapping(self):
        num_frame_in_refimgs = len(self.refimgs)
        num_frame_in_refvideos = []
        for i in range(len(self.refvideo_list)):
            elem = self.refvideo_list[i]
            if isinstance(elem, list):
                num_frame_in_refvideos.append(len(elem))
            else:
                num_frame_in_refvideos.append(elem.shape[0])
        assert sum(num_frame_in_refvideos) == num_frame_in_refimgs, f"Number of frames in refimgs {num_frame_in_refimgs} should match refvideos {num_frame_in_refvideos}"
        frame_vid_mapping = {}
        vid_idx = 0
        for i in range(len(self.refimgs)):
            while num_frame_in_refvideos[vid_idx] == 0:
                vid_idx += 1
            frame_vid_mapping[i] = vid_idx
            num_frame_in_refvideos[vid_idx] -= 1
        return frame_vid_mapping
                
    @torch.no_grad()
    def render_video_deform(self):
        frame_video_mapping = self.check_deform_sanity_and_mapping()
        frames = []
        for i in range(len(self.refimgs)):
            frame = self.refimgs[i]
            cam = self.refcams[i]
            vid_idx = frame_video_mapping[i]
            w2c = cam.getW2C()
            w2c = torch.tensor(w2c[None]).to(self.device)
            render_rgb, _, _, info = self._render_w2c_deform(cam, w2c, vid_idx)
            frames.append(render_rgb)
        return frames
    

    def train_deform(self, iters=None, confidence=None, enable_densification=False, enable_cam=None):
        ''' confidence: [N,H,W] tensor, same as refimgs, confidence map for the loss function '''
        frame_video_mapping = self.check_deform_sanity_and_mapping()
        refimg_shape = self.refimgs[0].shape
        N = len(self.refimgs)
        confidence = torch.ones((N,)+refimg_shape) if confidence is None else confidence
        assert confidence.shape[0] == N and confidence.shape[1:] == refimg_shape,\
              f"Confidence shape {confidence.shape} should match refimgs shape {N},{refimg_shape}"
        iters = self.opt.iters if iters is None else iters

        if enable_densification:
            self.enable_densification(iters)
        # disable densification for simple deform not support
        self.disable_densification()

        pbar = tqdm.tqdm(range(1, iters), desc="Training", leave=True)
        strategy = self.strategy
        print(strategy)

        if enable_cam is None:
            enable_cam = self.opt.enable_cam
        if enable_cam:
            self.pose_adjust = CameraOptModule(len(self.refimgs)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(self.pose_adjust.parameters(), lr=self.opt.cam_lr, weight_decay=self.opt.cam_rg)
            ]

        # self.deform_net = deform_network().to(self.device)
        # self.deform_optimizers = [
        #     torch.optim.Adam(self.deform_net.get_mlp_parameters(), lr=self.opt.deforma_mlp_lr),
        #     torch.optim.Adam(self.deform_net.get_grid_parameters(), lr=self.opt.deform_grid_lr)
        # ]

        self.deform_net = SimpleDeform(len(self.refvideo_list), self.GS.rgb.shape[0]).to(self.device)
        self.deform_optimizers = [
            torch.optim.Adam(self.deform_net.parameters(), lr=self.opt.deform_mlp_lr)
        ]

        param_dict = torch.nn.ParameterDict(self.GS.getgsplat_dict())
        self._init_optimizer_dict(param_dict=param_dict)
        strategy.check_sanity(param_dict, self.optimizer_dict)
        state = strategy.initialize_state()
        self.default_probability()

        for iter in pbar:
            frame_idx = np.random.choice(N, 1, p=self.probability)[0]
            vid_idx = frame_video_mapping[frame_idx]
            frame = self.refimgs[frame_idx]
            cam = self.refcams[frame_idx]
            
            w2c = cam.getW2C()
            if enable_cam:
                w2c = self.pose_adjust(torch.tensor(w2c[None]).to(self.device), torch.tensor([frame_idx]).to(self.device))
            else:
                w2c = torch.tensor(w2c[None]).to(self.device)

            render_rgb, _, _, info = self._render_w2c_deform(cam, w2c, vid_idx)

            loss_rgb = self.rgb_lossfunc(render_rgb, frame)
            strategy.step_pre_backward(param_dict, self.optimizer_dict, state, iter, info)
            loss = loss_rgb
            pbar.set_postfix({"Loss": loss.item()})
            loss.backward()  
            strategy.step_post_backward(param_dict, self.optimizer_dict, state, iter, info)
            self.GS.setgsplat_dict(param_dict)
            for optimizer in self.optimizer_dict.values():
                optimizer.step()
                optimizer.zero_grad()
            if enable_cam:
                for optimizer in self.pose_optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            for optimizer in self.deform_optimizers:
                optimizer.step()
                optimizer.zero_grad()
        refined_scene = self.GS
        return refined_scene

    def train_with_mask(self, iters=None, confidence=None, enable_densification=False):
        ''' confidence: [N,H,W] tensor, same as refimgs, confidence map for the loss function '''
        refimg_shape = self.refimgs[0].shape
        N = len(self.refimgs)
        confidence = torch.ones((N,)+refimg_shape) if confidence is None else confidence
        assert confidence.shape[0] == N and confidence.shape[1:] == refimg_shape,\
              f"Confidence shape {confidence.shape} should match refimgs shape {N},{refimg_shape}"
        iters = self.opt.iters if iters is None else iters
        if enable_densification:
            self.enable_densification(iters)
        pbar = tqdm.tqdm(range(1,iters+1), desc="Training", leave=True)
        strategy = self.strategy
        print(strategy)
        param_dict = torch.nn.ParameterDict(self.GS.getgsplat_dict())
        self._init_optimizer_dict(param_dict=param_dict)
        strategy.check_sanity(param_dict, self.optimizer_dict)
        state = strategy.initialize_state()
        for iter in pbar:
            frame_idx = np.random.randint(0,len(self.refimgs))
            frame = self.refimgs[frame_idx]
            cam = self.refcams[frame_idx]
            mask = self.refmask[frame_idx]
            render_rgb, _, _, info = self._render(cam)

            render_img = torch.zeros_like(frame)
            render_img[mask] = render_rgb[mask]
            frame_img = torch.zeros_like(frame)
            frame_img[mask] = frame[mask]

            if iter % 300 == 0:
                Ulog().add_img(frame, "frame")
                Ulog().add_img(frame_img, "frame_img")
                Ulog().add_img(render_img, "render_img")
                Ulog().add_img(render_rgb, "render_rgb")

            loss_rgb = self.rgb_lossfunc(frame_img, render_img)
            strategy.step_pre_backward(param_dict, self.optimizer_dict, state, iter, info)
            loss = loss_rgb
            pbar.set_postfix({"Loss": loss.item()})
            loss.backward()  
            strategy.step_post_backward(param_dict, self.optimizer_dict, state, iter, info)
            self.GS.setgsplat_dict(param_dict)
            for optimizer in self.optimizer_dict.values():
                optimizer.step()
                optimizer.zero_grad()
        refined_scene = self.GS
        return refined_scene
    
    def print_transform_matrix(self):
        print(self.pose_adjust.get_all_transform())