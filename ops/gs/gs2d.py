import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import gsplat 
from torchvision.utils import save_image
from plyfile import PlyData, PlyElement
from typing import Literal

from ops.cam_utils import Mcam



class GaussianMgr2D():
    ''' a GaussianMgr contain gaussian 
        params and can be rendered with no gradients, param is stored in cpu tensor
        self._rgb        : [M,3*(1+deg)**2] tensor
        self.xyz        : [M,3] tensor
        self._scale      : [M,2] tensor
        self._opacity    : [M,1] tensor
        self.rotation   : [M,4] tensor
        Note that the rgb, scale, opacity are in deact space
    '''
    rgb_deact    = torch.logit
    scale_deact  = torch.log
    opacity_deact = torch.logit
    rgb_act    = torch.sigmoid
    scale_act  = torch.exp
    opacity_act = torch.sigmoid


    def __init__(self, ply_file_path = None, pts3d = None):
        ''' Note there is no clone inside!!
        input: list of GaussianMgr
        input: pts3d, coarse initailize
        pts_3d: [M, 14] np.numpy|tensor xyz + rgb + opacity + scale + rotation
        pts_3d: list of above things
        pts_3d: list of GaussianMgr
        '''
        
        if ply_file_path is not None:
            self.load_ply(ply_file_path)
            print(f"{self.xyz.shape[0]} points loaded")
        elif pts3d is not None:

            if isinstance(pts3d, torch.Tensor):
                tensor_data = pts3d.detach().cpu()
                self.set_pts3d(tensor_data)
            elif isinstance(pts3d, np.ndarray):
                self.set_pts3d(pts3d)
            elif isinstance(pts3d, list):
                elem = pts3d[0]
                if isinstance(elem, torch.Tensor):
                    pts = np.concatenate([x.detach().cpu().numpy() for x in pts3d], axis=0)
                elif isinstance(elem, np.ndarray):
                    pts = np.concatenate(pts3d, axis=0)
                elif isinstance(elem, GaussianMgr):
                    pts = np.concatenate([x.get_pts3d() for x in pts3d], axis=0)
                else:
                    raise TypeError("Invalid elem type in pts3d, only support np or tensor")
                self.set_pts3d(pts)
            else:
                np_data = np.asarray(pts3d)
                self.set_pts3d(np_data)

        else:
            self._rgb        = None
            self.xyz        = None
            self._scale      = None
            self._opacity    = None
            self.rotation   = None

    @property
    def rgb(self):
        return GaussianMgr2D.rgb_act(self._rgb)
    
    @property
    def scale(self):
        return GaussianMgr2D.scale_act(self._scale)
    
    @property
    def opacity(self):
        return GaussianMgr2D.opacity_act(self._opacity)
    
    @rgb.setter
    def rgb(self, value):
        self._rgb = GaussianMgr.rgb_deact(value)
    
    @scale.setter
    def scale(self, value):
        self._scale = GaussianMgr.scale_deact(value)
    
    @opacity.setter
    def opacity(self, value):
        self._opacity = GaussianMgr.opacity_deact(value)
    

    def get_pts3d(self):
        ''' return [M, 14] tensor '''
        tensor_data = torch.cat([self.xyz, self._rgb, self._opacity, self._scale, self.rotation], dim=1)
        return tensor_data
    
    
    def set_pts3d(self, tensor_data):
        '''recover from [M, 14] tensor or np.ndarray'''
        if isinstance(tensor_data, np.ndarray):
            tensor_data = torch.from_numpy(tensor_data)
        tensor_data = tensor_data.detach().cpu()
        pack = self._unpack_data(tensor_data)
        self.xyz, self._rgb, self._opacity, self._scale, self.rotation = pack

    @staticmethod
    def _unpack_data(data):
        total_len = data.shape[1]
        sh_len = total_len - 10
        sh = np.sqrt(total_len // 3) - 1

        xyz        = data[:, :3]
        rgb        = data[:, 3:3+sh_len]
        opacity    = data[:, -7:-6]
        scale      = data[:, -6:-4]
        rotation   = data[:, -4:]
        return xyz, rgb, opacity, scale, rotation

    @staticmethod
    def _construct_dtype():
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(2):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        dtype_full = [(attribute, 'f4') for attribute in l]
        return dtype_full

    def save_ply(self, path, flip = True):
        ''' flip is to align with superSplat'''
    
        dtype_full = self._construct_dtype()
        elements = np.empty(self.xyz.shape[0], dtype=dtype_full)
        attributes = self.get_pts3d()
        if flip:
            attributes[:,[0,1]] = -attributes[:,[0,1]]
        attributes = attributes.cpu().numpy()
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
        print(f"total {self.xyz.shape[0]} points saved to {path}")

    def load_ply(self, path, flip = True):
        ''' flip is to align with superSplat'''
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]



        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        if len(extra_f_names) > 0:
            sh_len = len(extra_f_names) + 3
            sh = np.sqrt(sh_len // 3) - 1
            print(f"Warning: Tring to load 3dgs with sh>0, extra len:{sh_len}, sh:{sh}")
            print(extra_f_names)

        sh_len = len(extra_f_names) + 3
        features_dc = np.zeros((xyz.shape[0], sh_len))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        for idx, fname in enumerate(extra_f_names):
            features_dc[:, idx + 3] = np.asarray(plydata.elements[0][fname])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        pts3d = np.concatenate([xyz, features_dc, opacities, scales, rots], axis=1)
        pc = torch.from_numpy(pts3d).float()
        # assert 0
        # pc = plydata['vertex'].data
        # pc = torch.tensor(pc.tolist(), device="cpu", dtype=torch.float32)

        self.set_pts3d(pc)
        if flip:
            self.xyz[:,[0,1]] = -self.xyz[:,[0,1]]
        return self


    def _coarse_init_scale(self, cam:Mcam):
        cam_origin = cam.T
        xyz_np = self.xyz
        depth = np.linalg.norm(xyz_np - cam_origin, axis=1)
        self._scale = depth / cam.f / np.sqrt(2) 
        self._scale = self._scale[:,None].repeat(3,-1)


    def init_from_pts(self, pts, mode:Literal["knn","fixed","cam"] = "knn",
                       cam = None, scale = None):
        '''this function will have basic initial 
        property for gaussian segment, 
        mode = "knn" | "fixed" | "cam"
        mode = "knn": use knn to init scale
        mode = "fixed": use fixed scale
        mode = "cam": use depth / f to init scale

        pts: [M,6]  np.numpy xyz + rgb 
        '''
        valid_modes = ["knn", "fixed", "cam"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Expected one of {valid_modes}")

        pts = self._start_init(pts)
        M = pts.shape[0]
        self._init_basic(pts)
        if mode == "knn":
            from simple_knn._C import distCUDA2
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pts[:,:3])).float().cuda()), 1e-6)
            self._scale = torch.sqrt(dist2)[...,None].repeat(1, 3).cpu().numpy()
        elif mode == "fixed":
            assert scale is not None, "scale should be given in fixed mode"
            self._scale = np.ones((M,3)) * scale
        elif mode == "cam":
            assert cam is not None, "cam should be given in cam mode"
            self._coarse_init_scale(cam)
        self.rotation = np.zeros((M,4))
        self.rotation[:,0] = 1.
        self._finish_init()
        return self
    
    def _start_init(self, pts):
        '''check pts input; convert pts to [M,6] np.numpy'''
        M, C = pts.shape
        assert C == 6 or C == 3, "init pts should be [M,6] or [M,3], got {}".format(pts.shape)
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        elif isinstance(pts, np.ndarray):
            pass
        else:
            raise TypeError("pts should be tensor or np.ndarray")

        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.ones((M, 3))], axis=1)
        return pts

    
    def _finish_init(self):
        ''' turn to cpu tensor and fix act/deact
        '''
        def ten(x):
            return torch.from_numpy(x.astype(np.float32))
        
        self._rgb = ten(self._rgb)
        self.xyz = ten(self.xyz)
        self._scale = ten(self._scale)
        self._opacity = ten(self._opacity)
        self.rotation = ten(self.rotation)

        self._rgb = GaussianMgr.rgb_deact(self._rgb)
        self._scale = GaussianMgr.scale_deact(self._scale)
        self._opacity = GaussianMgr.opacity_deact(self._opacity)

        return self


    def _init_basic(self, pts):
        '''do basic init for rgb, xyz and opacity'''
        M = pts.shape[0]
        self.xyz = pts[:,:3]
        self._rgb = pts[:,3:]
        self._opacity = np.ones((M, 1)) * 0.8


    def render(self, cam:Mcam, save_path = None):
        ''' return [HW3] tensor rgb, [HW] tensor dpt, [HW] tensor alpha
        '''
        param = self.get_pts3d()
        render_rgb, render_dpt, render_alpha, render_normal, *_ = GaussianMgr2D._render(param, cam)
        if save_path is not None:
            save_image(render_rgb.permute(2,0,1)[None], save_path)
        return render_rgb, render_dpt, render_alpha

    @staticmethod
    def _render(param, cam:Mcam, background_color = (0,0,0)):
        '''
        param: [M, 10+3(1+sh)**2] tensor xyz + rgb + opacity + scale + rotation
        Note: rgb, opacity, scale should be in deact space 

        return: render_rgb[HW3], render_dpt[HW], render_alpha[HW]
        '''
        device = 'cuda' 

        background = torch.empty((1,3), dtype=torch.float32, device=device)
        background[:,0:3] = torch.tensor(background_color,device=device)

        param = param.to(device)
        xyz, rgb, opacity, scale, rotation = GaussianMgr2D._unpack_data(param)

        # seperately handle rgb
        sh_degree = None
        if rgb.shape[1] != 3:
            sh_degree = int(np.sqrt(rgb.shape[1] // 3) - 1)
            rgb = rgb.reshape(-1, (sh_degree+1)**2, 3)
            print(rgb.shape)
            print(background.shape)
        else:
            rgb = GaussianMgr2D.rgb_act(rgb)


        scale     = GaussianMgr2D.scale_act(scale)
        # add a new channel to align with gsplat
        scale = torch.cat([scale, torch.ones_like(scale)[:,[0]]*0.8], dim=1)

        rotation  = F.normalize(rotation,dim=1)
        opacity   = GaussianMgr2D.opacity_act(opacity)

        # gsplat need [M,] for opacity
        opacity   = opacity.squeeze(-1)

        H, W = cam.H, cam.W

        intrinsic = torch.from_numpy(cam.getK()).to(device)
        extrinsic = torch.from_numpy(cam.getW2C()).to(device)
        # OpenGL to OpenCV
        extrinsic[[1,2],:] = -extrinsic[[1,2],:]
        render_out,render_alpha,render_normal,surf_normal,distort,medium,meta = \
        gsplat.rendering.rasterization_2dgs(means = xyz,
                                                scales    = scale,
                                                quats     = rotation,
                                                opacities = opacity,
                                                colors    = rgb,
                                                Ks        = intrinsic[None],
                                                viewmats  = extrinsic[None],
                                                width     = W, 
                                                height    = H, 
                                                packed    = False,
                                                sh_degree = sh_degree,
                                                near_plane= 0.01,
                                                render_mode="RGB+ED",)
                                                #backgrounds=background) # render: 1*H*W*(3+1)
        render_out  = render_out.squeeze() # result: H*W*(3+1)
        render_rgb  = render_out[:,:,0:3]
        render_dpt  = render_out[:,:,-1]
        return render_rgb, render_dpt, render_alpha, render_normal, surf_normal, distort, medium

