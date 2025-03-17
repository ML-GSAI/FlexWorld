import sys

import trimesh
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras
from torchvision.utils import save_image
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import torch
from torch import nn

from ops.cam_utils import orbit_camera, Mcam
from copy import copy,deepcopy
import einops
from typing import Literal
from ops.utils.general import *

class PointsRenderer_Depth(nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer

    def to(self, device):
        self.rasterizer = self.rasterizer.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        zs = fragments.zbuf[:, :, :, 0] # [N, H, W]
        print(zs.shape)

        images = zs.unsqueeze(-1)

        return images


class PointsRenderer_Alpha(nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer

    def to(self, device):
        self.rasterizer = self.rasterizer.to(device)
        return self
    
    def forward_fast(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        masks = torch.logical_not(fragments.idx[:,:,:,0] < 0)
        images = masks.float().unsqueeze(-1)

        return images

    def forward_slow(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists
        alphas = 1 - dists2 / (r * r)

        masks = dists2 < 0
        alphas[masks] = 0.0
        alpha = torch.sum(alphas, dim=-1, keepdim=True).float()
        alpha = torch.clamp(alpha, 0, 1)
        images = alpha

        return images
    
    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        return self.forward_slow(point_clouds, **kwargs)


def setup_renderer(cameras, image_size, render_class = PointsRenderer):
    # Define the settings for rasterization and shading.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius = 0.01,
        points_per_pixel = 10,
        bin_size = 0
    )

    renderer = render_class(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'renderer': renderer}

    return render_setup



@dataclass
class Bbox:
    center: np.ndarray = field(default_factory= lambda :np.zeros(3))
    size: np.ndarray = field(default_factory= lambda :np.ones(3))


class PcdMgr():
    _render_backend = "pytorch3d"

    def __init__(self, ply_file_path = None, pts3d = None, diff_tensor = None):
        # pts3d: [N, 6], xyz,rgb  
        # rgb is in range [0,1]
        # diff_tensor is for rendering only, pts3d is for point cloud operation
        if ply_file_path is not None:
            point_cloud = o3d.io.read_point_cloud(ply_file_path)
            pts = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors)
            self.pts = np.concatenate((pts, colors), axis=1)
            print(f"{pts.shape[0]} points loaded")
        elif pts3d is not None:
            if isinstance(pts3d, torch.Tensor):
                self.pts = to_numpy(pts3d)
            elif isinstance(pts3d, list):
                elem = pts3d[0]
                if isinstance(elem, torch.Tensor):
                    assert len(elem.shape) > 1 and elem.shape[-1] == 6, "Invalid shape for pts3d, shape is {}".format(elem.shape)
                    pts3d = [to_numpy(x).reshape(-1, 6) for x in pts3d]
                    self.pts = np.concatenate(pts3d, axis=0)
                elif isinstance(elem, np.ndarray):
                    self.pts = np.concatenate(pts3d, axis=0)
                else:
                    raise TypeError("Invalid elem type in pts3d, only support np or tensor")
            else:
                self.pts = np.asarray(pts3d)
            C = self.pts.shape[-1]
            assert C == 3 or C == 6, "Invalid shape for pts3d, shape is {}".format(self.pts.shape)
            self.pts = self.pts.reshape(-1, C)
        elif diff_tensor is not None:
            self.diff_tensor = diff_tensor
            self.pts = diff_tensor

    def add_mark_point(self, pos, num=10000, radius=0.005, color=(1,0,0)):
        # for debug
        dx = (np.random.rand(num, 3) - 0.5) * radius
        newpts = pos + dx
        newpts = np.concatenate((newpts, np.tile(color, (num,1))), axis=1)

        self.pts = np.concatenate((self.pts, newpts), axis=0)

    def transform_objects(self, flip_y=True):
        # flip y if directly get result from instant-mesh, if read from obj, set flip_y=False
        rotation_matrix = np.array([[0, 0, 1],
                            [1, 0, 0],
                            [0, -1, 0]])
        self.pts[:,:3] =  self.pts[:, :3] @ rotation_matrix
        if flip_y:
            self.pts[:,1] = -self.pts[:,1]
        box = self.find_bbox(self.pts[:,:3]) # [!] duplicate here
        self.pts[:,:3] = 1/box.size[1] * self.pts[:,:3] # scale using y

    @classmethod
    def set_default_render_backend(cls, backend:Literal["pytorch3d", "gs"]):
        cls._render_backend = backend
    @classmethod
    def get_default_render_backend(cls):
        return cls._render_backend
    

    def render(self, cam:Mcam, save_path=None, mask=False, depth=False, backends:Literal["pytorch3d", "gs"]=None):
        ''' return rendered image with shape [1,3,H,W]
        pts: [N, 6], xyz,rgb   np.ndarray or torch.tensor
        if backends is None, use class default
        '''
        if hasattr(self, "diff_tensor"):
            pts = self.diff_tensor
        else:
            pts = self.pts
        if backends is None:
            backends = self._render_backend
        if backends == "gs":
            img = PcdMgr._render_using_gs(pts, cam, mask, depth)
        elif backends == "pytorch3d":
            img = PcdMgr._render(pts, cam, mask, depth)
        else:
            raise ValueError(f"Invalid backend {backends}")
        if save_path is not None:
            save_image(img, save_path)
        return img
    
    @staticmethod
    def _render_using_gs(pts, cam: Mcam, gs, mask=False, depth=False):
        ''' return rendered image with shape [1,3,H,W]
        pts: [N, 6], xyz,rgb   np.ndarray or torch.tensor
        mask=0 if no points, 
        in mask return [1,1,H,W]
        '''
        from ops.gs.base import GaussianMgr
        gs = GaussianMgr()
        gs.init_from_pts(pts, mode="fixed", scale=0.0003, opacity=0.95)
        rgb, depth_img, alpha_img, *_ = gs.render(cam)
        # to match with original design
        if mask:
            return einops.rearrange(alpha_img, 'h w -> 1 1 h w')
        elif depth:
            return einops.rearrange(depth_img, 'h w -> 1 1 h w')
        else:
            return einops.rearrange(rgb, 'h w c -> 1 c h w')
    
    @staticmethod
    def _render(pts, cam: Mcam, mask=False, depth=False):
        ''' return rendered image with shape [1,3,H,W]
        pts: [N, 6], xyz,rgb   np.ndarray or torch.tensor
        mask=0 if no points, 
        in mask return [1,1,H,W]
        '''
        # [!] following "cuda" should be changed to device later
        # camera here is OpenGL camera
        def t(x):
            return [x]
        def ten(x):
            return torch.tensor(x, dtype=torch.float, device='cuda').unsqueeze(0)
        if isinstance(pts, np.ndarray):
            pts = torch.tensor(pts,dtype=torch.float,device='cuda')
        point_cloud1 = Pointclouds(points=[pts[:,:3]],features=[pts[:, 3:]])
        R, T = cam.R.copy(), cam.T.copy()
        # OpenGL -> pytorch3d
        R[:,[0,2]] = -R[:,[0,2]]
        T = -R.T @ T
        if mask:
            render_class = PointsRenderer_Alpha
        elif depth:
            render_class = PointsRenderer_Depth
        else:
            render_class = PointsRenderer


        ccc=PerspectiveCameras(focal_length=t(cam.f), principal_point=t(cam.c), in_ndc=False,
                                image_size=t((cam.H, cam.W)), R=ten(R), T=ten(T),device='cuda')

        render_setup = setup_renderer(ccc, (cam.H, cam.W), render_class)
        renderer = render_setup['renderer']
        img=renderer(point_cloud1).permute(0,3,1,2)
        return img

        
    @staticmethod
    def find_bbox(pts, z_filter = False):
        def m(nd):
            return (nd.max() + nd.min())/2
        bbox = Bbox()
        
        if z_filter:
            z_coords = pts[:, 2]
            lower_bound = np.percentile(z_coords, 5)
            front_pt = pts[(z_coords >= lower_bound)]

            bbox.center[2] = m(front_pt[:, 2])
            pts = front_pt
        else:
            bbox.center[2] = m(pts[:, 2])

        bbox.center[0], bbox.center[1] = m(pts[:, 0]), m(pts[:, 1])
        bbox.size = 2*(pts[:, :3].max(axis=0) - bbox.center)

        return bbox
    
    def remove_using_traj(self, traj:Mcam, threshold=0.05):
        for cam in traj:
            distances = np.linalg.norm(self.pts[:, :3] - cam.T, axis=1)
            close_points_mask = distances <= threshold
            self.pts = self.pts[~close_points_mask]
            print("remove", sum(close_points_mask), "points")


    def remove_outliers(self, nb_points=16, radius=0.003):
        """
        points_rgb: [N, 6] 的 numpy 数组 (x, y, z, r, g, b)
        返回去除散点后的 [M, 6] 的 numpy 数组
        """
        # 创建 open3d 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(self.pts[:, 3:6])

        # 去除散点
        _, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

        # 获取去除散点后的结果
        inlier_pcd = pcd.select_by_index(ind)
        inlier_points = np.hstack((np.asarray(inlier_pcd.points), np.asarray(inlier_pcd.colors)))
        print(f"原始点云数量: {self.pts.shape[0]}, 去除散点后数量: {inlier_points.shape[0]}")
        self.pts = inlier_points

    def remove_outliers_near(self, nb_points=16, radius=0.003, distance_threshold=0.3):
        """
        去除离原点近的点
        points_rgb: [N, 6] 的 numpy 数组 (x, y, z, r, g, b)
        返回去除散点后的 [M, 6] 的 numpy 数组
        """
        # 过滤距离原点超过 distance_threshold 的点
        distances = np.linalg.norm(self.pts[:, :3], axis=1)
        close_points_mask = distances <= distance_threshold
        close_points = self.pts[close_points_mask]

        # 创建 open3d 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(close_points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(close_points[:, 3:6])

        # 去除散点
        _, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

        # 获取去除散点后的结果
        inlier_pcd = pcd.select_by_index(ind)
        inlier_points = np.hstack((np.asarray(inlier_pcd.points), np.asarray(inlier_pcd.colors)))

        # 保留距离原点超过 distance_threshold 的点
        far_points = self.pts[~close_points_mask]

        print(f"原始点云数量: {self.pts.shape[0]}, 去除散点后数量: {inlier_points.shape[0] + far_points.shape[0]}")
        
        # 合并去除散点后的近点和远点
        self.pts = np.vstack((inlier_points, far_points))

    def save_ply(self, save_path):
        pc = trimesh.PointCloud(self.pts[:,:3], colors=self.pts[:,3:])

        # Define a default normal, e.g., [0, 1, 0]
        default_normal = [0, 1, 0]

        # Prepare vertices, colors, and normals for saving
        vertices = pc.vertices
        colors = pc.colors
        normals = np.tile(default_normal, (vertices.shape[0], 1))

        # Construct the header of the PLY file
        header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

        # Write the PLY file
        with open(save_path, 'w') as ply_file:
            ply_file.write(header)
            for vertex, color, normal in zip(vertices, colors, normals):
                ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                    vertex[0], vertex[1], vertex[2],
                    int(color[0]), int(color[1]), int(color[2]),
                    normal[0], normal[1], normal[2]
                ))


    def add_pts(self, pts):
        '''pts: [..., 6] add new points to self.pts'''
        if isinstance(pts, torch.Tensor):
            pts = pts.cpu().detach().numpy()
        elif isinstance(pts, self.__class__):
            pts = pts.pts
        elif isinstance(pts, list):
            pts = np.concatenate(pts, axis=0)
        assert pts.shape[-1] == 6, f"pts should have 6 columns, got {pts.shape}"
        pts = pts.reshape(-1, 6)
        self.pts = np.concatenate([self.pts, pts], axis=0)

    def get_pts(self):
        '''return pts inside with [M, 6]'''
        return self.pts
    
    def clone(self):
        return PcdMgr(pts3d=self.pts.copy())
    
