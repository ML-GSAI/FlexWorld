from typing import List
import numpy as np
import torch
from ops.PcdMgr import PcdMgr, Bbox
from ops.cam_utils import Mcam
from dataclasses import field,dataclass
from torch import optim
from torchvision.utils import save_image

@dataclass
class ObjOpertion:
    dx:np.ndarray = field(default_factory=lambda:np.zeros(3))
    scale:np.ndarray = field(default_factory=lambda:np.ones(3))
    rotation:np.ndarray = field(default_factory=lambda:np.zeros(1))
    translation:np.ndarray = field(default_factory=lambda:np.zeros(3))



def pts2plane(pts):
    """
    use ransac
    """
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression
    
    point_cloud =pts[:,:3]
    avg_color=pts[:,3:].mean(axis=0)
    X = point_cloud[:, :2]  # (N, 2)
    y = point_cloud[:, 2]   # (N,)
    
    model = LinearRegression()

    ransac = RANSACRegressor(model, min_samples=500, residual_threshold=0.05, max_trials=1000)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    coef = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_
    print("平面方程: z = {:.3f} * x + {:.3f} * y + {:.3f}".format(coef[0], coef[1], intercept))
    
    x_range = np.linspace(point_cloud[:, 0].min(), point_cloud[:, 0].max(), 300)
    y_range = np.linspace(point_cloud[:, 1].min(), point_cloud[:, 1].max(), 300)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = coef[0] * xx + coef[1] * yy + intercept
    new_points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    
    check_dist = True
    if check_dist:
        print("原始点数：", new_points.shape[0])
        from scipy.spatial import cKDTree
        tree = cKDTree(point_cloud)
        distances, _ = tree.query(new_points, k=1)
        distance_threshold = 3e-4  # 根据需要调整阈值
        mask = distances > distance_threshold
        print("剔除距离小于阈值的点，剩余点数：", mask.sum())
        new_points = new_points[mask]
    
    num_new_points = new_points.shape[0]
    new_colors = np.tile(avg_color, (num_new_points, 1))
    
    # 合并 new_points 和 new_colors
    new_points_with_color = np.hstack((new_points, new_colors))
    
    print(new_points_with_color.shape)
    
    final_pts=np.concatenate((pts[inlier_mask], new_points_with_color),axis=0)
    # final_pts=np.concatenate((pts, new_points_with_color),axis=0)
    
    print(final_pts.shape)
    
    return final_pts


class ObjectMgr():
    def __init__(self):
        self.pms = [] # list of [W,H,6] xyz+rgb
        self.masks = []   # list of [W,H] 0 will be removed
        self.pts = [] # [M,6], only for objects
        self.scene_pts = [] # [M,6], only for scenes
        self.plane_pts = []
        self.obj_masks = []
        self.obj_operations : List[ObjOpertion] = [] 
        self.plane_masks = []
        self.bboxes = []
        self.bg_pm = None
    
    def add_pms(self, pms, masks=None):
        if isinstance(pms, torch.Tensor):
            pms = pms.cpu().numpy()
        if masks is None:
            masks = np.ones_like(pms[...,0],dtype=np.bool8)
        self.pms.append(pms)
        self.masks.append(masks)

    def add_objects(self, objmask, newpts, pm_idx=0):
        # pm_idx: objmask corresponding to which mask in self.masks
        assert objmask.shape == self.masks[pm_idx].shape
        self.obj_masks.append(objmask)
        self.pts.append(newpts)
        self.masks[pm_idx][objmask] = 0
        selected_pts = self.pms[pm_idx][objmask]
        print(f"add {newpts.shape[0]} pts , original has {selected_pts.shape[0]} points")
        self.bboxes.append(self.find_bbox_with_pts(selected_pts, z_filter=True))
        
    def add_background(self, bgpts):
        self.bg_pm = bgpts
        
    def add_scenes(self, newpts, cam:Mcam):
        newpts = newpts @ cam.getC2W()[:3, :3].T + cam.getC2W()[:3, 3].T
        self.scene_pts.append(newpts)
    
    def add_planes(self, planemask, pm_idx=0):
        # pm_idx: planemask corresponding to which mask in self.masks
        assert planemask.shape == self.masks[pm_idx].shape
        self.plane_masks.append(planemask)
        newpts = pts2plane(self.mask2pts(planemask))
        self.plane_pts.append(newpts)
        selected_pts = self.pms[pm_idx][planemask]
        print(f"add {newpts.shape[0]} pts")
        # self.bboxes.append(self.find_bbox_with_pts(selected_pts, z_filter=True))
 
        
    def mask2pts(self,objmask, pm_idx=0):
        selected_pts = self.pms[pm_idx][objmask]
        return selected_pts

    @staticmethod
    def find_bbox_with_pts(pts, z_filter = False):
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
    
    def placeAllobj(self, pmidx=0):
        if len(self.obj_masks) == 0:
            return np.empty((0,6))
        for i in range(len(self.obj_masks)):
            self.placeobj(i, pmidx)
        pts_obj_list = self.optimize(pmidx)
        obj_pts = torch.cat(pts_obj_list, dim=0).cpu().detach().numpy()
        # pts_obj_list = [self.placeobj(i, pmidx) for i in range(len(self.obj_masks))]
        # obj_pts = np.concatenate(pts_obj_list, axis=0)
        return obj_pts
        
    def placeobj(self, idx, pmidx=0):
        # [!] this function will pose inital scale only now, maybe this should be moved to optimization later
        pts = self.pts[idx]
        bbox_obj = self.find_bbox_with_pts(pts[:,:3])
        pts_origin = self.pms[pmidx][self.obj_masks[idx]]
        bbox_origin = self.bboxes[idx]
        print(bbox_origin)
        
        def find_lf_point(pts, bbox):
            # find the lowest frontest point
            weighted_values = 0.5 * (pts[:, 1]) + 0.5 * (-pts[:, 2])
            min_index = np.argmin(weighted_values)
            return pts[min_index][:3]
        
        def find_filtered_center_point(pts, bbox):
            center_z = bbox.center[2]
            filtered_pts = pts[pts[:, 2] > center_z]
            center_pts = filtered_pts.mean(axis=0)[:3]
            return center_pts
        
        def find_center_point(pts, bbox):
            center_pts = pts.mean(axis=0)[:3]
            return center_pts

        # first scale then find feature point, should scale first
        # [!] this should be considered carefully later, for thin objects will fail
        pts[:,:3] *= bbox_origin.size[1]

        pts_coord = find_center_point(pts, bbox_obj)
        origin_coord = find_center_point(self.pms[pmidx][self.obj_masks[idx]],bbox_origin)
        
        dx = origin_coord - pts_coord
        op = ObjOpertion(dx=dx)
        self.obj_operations.append(op)
        self.pts[idx] = pts.copy()
        pts[:,:3] += dx
        return pts
    
    def placeplane(self,idx):
        pts = self.plane_pts[idx]
        return pts
    
    def mask2pts(self,objmask, pm_idx=0):
        selected_pts = self.pms[pm_idx][objmask]
        return selected_pts

    def construct_pcd(self):
        pts_pm = np.concatenate([self.pms[i][m] for i,m in enumerate(self.masks)])

        pts_obj = self.placeAllobj()
        # if len(self.plane_masks) > 0:
        #     pts_plane = np.concatenate([self.placeplane(m) for m in range(len(self.plane_masks))])
        # else:
        #     pts_plane = np.empty((0,6))
        # pts = np.concatenate([pts_pm, pts_obj, pts_plane])
        if self.bg_pm is not None:
            pts_bg = self.bg_pm
            pts = np.concatenate([pts_pm, pts_obj, pts_bg])
        else:
            pts = np.concatenate([pts_pm, pts_obj])
        return PcdMgr(pts3d=pts)
    
    def optimize(self, pmidx=0):
        M = len(self.obj_masks)
        def T(x):
            return torch.tensor(x, dtype=torch.float32, device="cuda")
        
        pts_tensor_list = [T(self.pts[i]) for i in range(M)]
        pts_background = self.pms[pmidx][self.masks[pmidx]]
        
        rots = T(np.stack([op.rotation for op in self.obj_operations])).requires_grad_()
        trans = T(np.stack([op.translation for op in self.obj_operations])).requires_grad_()
        scales = T(np.stack([op.scale for op in self.obj_operations])).requires_grad_()
        dxs = T(np.stack([op.dx for op in self.obj_operations]))
        pts_background = T(pts_background)


        optimizer = optim.Adam([
            {'params': rots, 'lr': 0.005},
            {'params': scales, 'lr': 0.002},
            {'params': trans, 'lr': 0.0005}
        ])

        def rot_pts(pts_tensor, rotation_y, scale, dx, translation):
            # 施加绕 y 轴的旋转
            rotation_matrix = torch.eye(3, device="cuda")
            rotation_matrix[0, 0] = torch.cos(rotation_y)
            rotation_matrix[0, 2] = torch.sin(rotation_y)
            rotation_matrix[2, 0] = -torch.sin(rotation_y)
            rotation_matrix[2, 2] = torch.cos(rotation_y)
            pts_scaled = pts_tensor[:, :3] * scale
            pts_rotated = torch.matmul(pts_scaled, rotation_matrix) + dx + translation
            pts_rotated_col = torch.cat([pts_rotated, pts_tensor[:, 3:]], dim=1)
            return pts_rotated_col
        
        pts_rotated = [torch.empty_like(pts_tensor_list[i]) for i in range(M)]
        pts_rotated.append(pts_background)
        default_cam = Mcam()
        img_raw = PcdMgr(pts3d=self.pms[pmidx].reshape((-1, 6))).render(default_cam)
        pts_in_raw = [self.pms[pmidx][self.obj_masks[i]] for i in range(M)]
        img_raw_nobg = PcdMgr(pts3d=pts_in_raw).render(default_cam)
        img_raw_nobg_msk = PcdMgr(pts3d=pts_in_raw).render(default_cam, mask=True)

        for epoch in range(200):
            optimizer.zero_grad()
            
            for i in range(M):
                pts_rotated[i] = rot_pts(pts_tensor_list[i], rots[i], scales[i], dxs[i], trans[i])
            
            pts_all = torch.cat(pts_rotated, dim=0)
            pts_all_nobg = torch.cat(pts_rotated[:-1], dim=0)
            img = PcdMgr(diff_tensor=pts_all).render(default_cam)
            img_nobg = PcdMgr(diff_tensor=pts_all_nobg).render(default_cam)
            img_nobg_msk = PcdMgr(diff_tensor=pts_all_nobg).render(default_cam, mask=True)
            loss = torch.nn.functional.mse_loss(img, img_raw) +  torch.nn.functional.mse_loss(img_nobg, img_raw_nobg) +  10*torch.nn.functional.mse_loss(img_nobg_msk, img_raw_nobg_msk)
            #loss = torch.nn.functional.mse_loss(img_nobg, img_raw_nobg)
            loss.backward()
            optimizer.step()
            if epoch == 0:
                save_image(img_nobg, f"./cache/render_1.png")
                save_image(img, f"./cache/render_0.png")
            if epoch % 10 == 0:
                save_image(img, f"./cache/render_2.png")
                save_image(img_nobg_msk, f"./cache/render_1.png")
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        print(f"scales:{scales}")
        print(f"trans:{trans}")
        print(f"rots:{rots}")

        def NP(x):
            return x.detach().cpu().numpy()
        scales = NP(scales)
        trans = NP(trans)
        rots = NP(rots)
        dxs = NP(dxs)

        self.obj_operations = [ObjOpertion(scale=scales[i], translation=trans[i], rotation=rots[i], dx=dxs[i]) for i in range(M)]

        return pts_rotated
