from torchvision.utils import save_image
import open3d as o3d
from PIL import Image
import numpy as np
import torch

from ops.PcdMgr import PcdMgr
from ops.cam_utils import orbit_camera
import torch.nn.functional as F
import os
import torchvision
from ops.cam_utils import Mcam, CamPlanner
from ops.utils.depth import refine_depth, depth2pcd_world, refine_depth_tmp
from copy import deepcopy
from scipy.ndimage import binary_dilation
import einops
from ops.utils.general import convert_PIL
from ops.gs.base import GaussianMgr
import imageio
from ops.utils.logger import Ulog

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

def get_input_dict(img_tensor,idx,dtype = torch.float32):
    """
        img_tensor: [-1,1]
    """
    print(img_tensor.shape)
    return {'img':F.interpolate(img_tensor.to(dtype), size=(288, 512), mode='bilinear', align_corners=False), 'true_shape': np.array([[288, 512]], dtype=np.int32), 'idx': idx, 'instance': str(idx), 'img_ori':img_tensor.to(dtype)}


def put_video_into_pcd_with_filter(pcd: PcdMgr, diffusion_results, cam_trajs, dust3r, opt, img_new_num = 3):
    img_new_inds = np.linspace(0, len(diffusion_results) - 1, img_new_num + 1, dtype=int)[1:]
    for ind in img_new_inds:
        img_new = diffusion_results[ind]
        pcd = put_image_into_pcd_with_filter(pcd, img_new, cam_trajs[ind], dust3r, opt)
    return pcd

def put_video_into_pcd(pcd: PcdMgr, diffusion_results, cam_trajs, dust3r, opt, img_new_num = 3):
    img_new_inds = np.linspace(0, len(diffusion_results) - 1, img_new_num + 1, dtype=int)[1:]
    for ind in img_new_inds:
        img_new = diffusion_results[ind]
        pcd, mask = put_image_into_pcd_mask(pcd, img_new, cam_trajs[ind], dust3r, opt)
    return pcd


def put_video_into_gs(gs: GaussianMgr, diffusion_results, cam_trajs, dust3r, opt, img_new_num = 3):
    img_new_inds = np.linspace(0, len(diffusion_results) - 1, img_new_num + 1, dtype=int)[1:]
    
    for ind in img_new_inds:
        img_new = diffusion_results[ind]
        gs = put_image_into_gs(gs, img_new, cam_trajs[ind], dust3r, opt)
    return gs

def put_video_into_gs_and_train(gs: GaussianMgr, diffusion_results, cam_trajs, dust3r, opt, train_config, img_new_num = 3):
    from ops.gs.train import GS_Train_Tool
    put_video_into_gs(gs, diffusion_results, cam_trajs, dust3r, opt, img_new_num)
    trainer = GS_Train_Tool(gs, train_config)
    trainer.add_trainable_video(diffusion_results, cam_trajs)
    refined_gs = trainer.train_with_cam(enable_densification=True)
    return refined_gs.getMgr()

def put_image_into_pcd_mask(pcd: PcdMgr, img_new, cam, dust3r, opt):
    """
        img: [H, W, 3] in [0, 1]
        img_depth: [H, W]
        known_depth: [H, W]

        return: [H, W, 6] pts3d 
    """
    new_img = dust3r._load_our_images(img_new)
    new_scene= dust3r.run_dust3r(new_img)
    estimated_depth = new_scene.get_depthmaps()[0].detach().cpu().numpy()

    col_img = new_img[0]['img'] * 0.5 + 0.5
    col_img = einops.rearrange(col_img, '1 c h w -> h w c')
    
    mask = pcd.render(cam,mask=True).squeeze().detach().cpu().numpy()
    img = pcd.render(cam).squeeze().detach().cpu().numpy()
    Ulog().add_img(img, "img")
    mask = (mask == 0)
    # we do some dilation to make sure the mask is large enough
    dilated_mask = binary_dilation(mask, iterations=1) 
    known_depth = pcd.render(cam, depth=True).squeeze().detach().cpu().numpy() 

    new_depth = refine_depth(known_depth, estimated_depth, dilated_mask)

    new_pm = depth2pcd_world(new_depth, cam)
    new_pts6d = np.concatenate([new_pm, col_img], axis=-1)
    pcdtmp = PcdMgr(pts3d=new_pts6d[dilated_mask])
    pcdtmp.remove_outliers_near()
    pcd.add_pts(pcdtmp.pts) 
    return pcd, dilated_mask


def put_image_into_pcd_with_filter(pcd: PcdMgr, img_new, cam, dust3r, opt):
    """
        img: [H, W, 3] in [0, 1]
        img_depth: [H, W]
        known_depth: [H, W]

        return: [H, W, 6] pts3d 
    """
    new_img = dust3r._load_our_images(img_new)
    new_scene= dust3r.run_dust3r(new_img)
    estimated_depth = new_scene.get_depthmaps()[0].detach().cpu().numpy()

    laplacian_kernel = torch.tensor([
        [[[ 0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]]]
    ], dtype=torch.float32, device="cuda")
    edges_target = F.conv2d(torch.tensor(estimated_depth, device="cuda").unsqueeze(0), laplacian_kernel, padding=1)
    img = torch.abs(edges_target[0,0,...]).cpu().numpy()
    mask = img > 0.05
    mask = torch.tensor(mask, device="cuda")
    kernel = torch.ones((3, 3), dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0)  # 定义膨胀核
    mask_dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel, padding=1).squeeze(0).squeeze(0) > 0
    mask_grad = mask_dilated.cpu().numpy()


    col_img = new_img[0]['img'] * 0.5 + 0.5
    col_img = einops.rearrange(col_img, '1 c h w -> h w c')
    
    mask = pcd.render(cam,mask=True).squeeze().detach().cpu().numpy()
    # we identify the mask where the gradient is large



    mask = (mask == 0)
    # we do some dilation to make sure the mask is large enough
    mask = binary_dilation(mask, iterations=25) 

    known_depth = pcd.render(cam, depth=True).squeeze().detach().cpu().numpy() 

    new_depth = refine_depth(known_depth, estimated_depth, mask)

    new_pm = depth2pcd_world(new_depth, cam)
    new_pts6d = np.concatenate([new_pm, col_img], axis=-1)

    mask = mask & mask_grad

    pcd.add_pts(new_pts6d[mask]) 
    return pcd

def put_image_into_pcd(pcd: PcdMgr, img_new, cam, dust3r, opt):
    """
        img: [H, W, 3] in [0, 1]
        img_depth: [H, W]
        known_depth: [H, W]

        return: [H, W, 6] pts3d 
    """
    new_img = dust3r._load_our_images(img_new)
    new_scene= dust3r.run_dust3r(new_img)
    estimated_depth = new_scene.get_depthmaps()[0].detach().cpu().numpy()

    col_img = new_img[0]['img'] * 0.5 + 0.5
    col_img = einops.rearrange(col_img, '1 c h w -> h w c')
    
    mask = pcd.render(cam,mask=True).squeeze().detach().cpu().numpy()
    mask = (mask == 0)
    # we do some dilation to make sure the mask is large enough
    mask = binary_dilation(mask, iterations=25) 
    known_depth = pcd.render(cam, depth=True).squeeze().detach().cpu().numpy() 

    new_depth = refine_depth(known_depth, estimated_depth, mask)

    
    new_pm = depth2pcd_world(new_depth, cam)
    new_pts6d = np.concatenate([new_pm, col_img], axis=-1)
    pcd.add_pts(new_pts6d[mask]) 
    return pcd



def put_image_into_gs(gs: GaussianMgr, img_new, cam, dust3r, opt):
    """
        img: [H, W, 3] in [0, 1]
        img_depth: [H, W]
        known_depth: [H, W]

        return: [H, W, 6] pts3d 
    """
    new_img = dust3r._load_our_images(img_new)
    new_scene= dust3r.run_dust3r(new_img)
    estimated_depth = new_scene.get_depthmaps()[0].detach().cpu().numpy()

    col_img = new_img[0]['img'] * 0.5 + 0.5
    col_img = einops.rearrange(col_img, '1 c h w -> h w c')
    
    _, known_depth, mask, _ = gs.render(cam)
    known_depth, mask = known_depth.squeeze().detach().cpu().numpy(), mask.squeeze().detach().cpu().numpy()
    mask = (mask < 0.5)
    # we do some dilation to make sure the mask is large enough
    mask = binary_dilation(mask, iterations=25) 

    try:
        new_depth = refine_depth(known_depth, estimated_depth, mask)
    except:
        new_depth = estimated_depth
        print("refine depth failed")

    new_pm = depth2pcd_world(new_depth, cam)
    new_pts6d = np.concatenate([new_pm, col_img], axis=-1)
    # new_gs = GaussianMgr().init_from_pts(new_pts6d[mask], mode="fixed", scale=3e-4)
    pcdtmp = PcdMgr(pts3d=new_pts6d[mask])
    # pcdtmp.remove_outliers_near()
    pcdtmp.remove_outliers()
    new_gs = GaussianMgr().init_from_pts(pcdtmp.pts, mode="fixed", scale=3e-4)
    gs.merge(new_gs)
    return gs


class Video_Tool():
    def __init__(self,opt, dust3r=None, type_diffusion=None, device='cuda', tmp=False):
        self.opt = opt
        
        if type_diffusion is None:
            type_diffusion = opt.type_diffusion
        
        if not tmp:
            if type_diffusion=='cogvideo':
                from ops.cogvideo import CogVideo
                self.video_diffusion = CogVideo(opt,device=device)
            elif type_diffusion=='viewcrafter':
                from ops.viewcrafter import ViewCrafter
                self.video_diffusion = ViewCrafter(opt,device=device)
            else:
                raise NotImplementedError(f"diffusion type {type_diffusion} not implemented")
            
        self.device = device
        self.dust3r = dust3r
        self.images = []
        self.idx = 0

    def only_run_cog(self, render_results, prompts = None):
        '''
        render_results: [T, H, W, 3], [0, 1]
        return: [T, H, W, 3], [0, 1]
        '''
        render_results = torch.stack(render_results,dim=0)
        render_results = render_results.permute(0,3,1,2)
        render_results = F.interpolate(render_results, size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        diffusion_results = self.video_diffusion.nvs_single_view(render_results, prompts) # [T, H, W, 3], [0, 1]
        return diffusion_results

        
    def run(self, cam_trajs, img_ref, pcd: PcdMgr, prompts = None, logger=True):
        """
            img_ref: [H, W, 3], range: [0, 255]
        """
        img_ref = torch.tensor(np.array(img_ref).astype(np.float32)/255.0).to(self.device)
                
        render_results = CamPlanner._render_video(pcd, traj = cam_trajs)
        render_mask_results = CamPlanner._render_video(pcd, traj = cam_trajs, mask=True)
        if logger:
            Ulog().add_video([frame.unsqueeze(-1) for frame in render_mask_results], "render_mask_results")
        render_results = torch.stack(render_results,dim=0)
        render_results = render_results.permute(0,3,1,2)
        render_results = F.interpolate(render_results, size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = img_ref
        # save_video(render_results, "cache/render.mp4")
        if logger:
            Ulog().add_video(render_results, "render_results")
        
        render_results = render_results.to(self.device)
        diffusion_results = self.video_diffusion.nvs_single_view(render_results, prompts) # [T, H, W, 3], [0, 1]
        # save_video(diffusion_results, "cache/diffusion.mp4")
        if logger:
            Ulog().add_video(diffusion_results, "diffusion_results")
        diffusion_results[0] = img_ref
        
        return diffusion_results
    
    def add_scene(self, cam_trajs, img_ref, pcd: PcdMgr, img_new_num = 3, prompts = None, input_ref =True):
        # generate novel image
        diffusion_results = self.run(cam_trajs, img_ref, pcd, prompts)
        img_new_inds = np.linspace(0, len(diffusion_results) - 1, img_new_num + 1, dtype=int)[1:]
        
        for ind in img_new_inds:
            img_new = diffusion_results[ind]
            pcd = put_image_into_pcd(pcd, img_new, cam_trajs[ind], self.dust3r, self.opt)
            pcd.remove_outliers()
            Ulog().add_ply(pcd, "pcd_tmp")
        
        if input_ref:
            img_last = img_ref
        else:
            img_last = (img_new.cpu().detach().numpy()*255).astype(np.uint8)
            
        return pcd, img_last , diffusion_results
    
    def add_scene_gs(self, cam_trajs, img_ref, gs_pipe, img_new_num = 3, add_train=True,prompts = None, input_ref =True):
        # generate novel image
        diffusion_results = self.run(cam_trajs, img_ref, gs_pipe.gs, prompts)
        img_new_inds = np.linspace(0, len(diffusion_results) - 1, img_new_num + 1, dtype=int)[1:]
        
        img_news, cam_news =[], []
        for ind in img_new_inds:
            img_new = diffusion_results[ind]
            gs = put_image_into_gs(gs_pipe.gs, img_new, cam_trajs[ind], self.dust3r, self.opt)
            gs_pipe.gs_update(gs)
            Ulog().add_ply(gs, "pcd_tmp")
            img_news.append(img_new)
            cam=cam_trajs[ind]
            cam_news.append(cam)
        img_news=torch.stack(img_news, dim=0)
        if add_train:
            gs_pipe.add_frame(diffusion_results, cam_trajs)
            # gs_pipe.add_frame(img_news, cam_news)
            gs_pipe.train(full_train=False)
            Ulog().add_ply(gs_pipe.gs, "pcd_tmp")
        
        if input_ref:
            img_last = img_ref
        else:
            img_last = (img_new.cpu().detach().numpy()*255).astype(np.uint8)
            
        return gs_pipe, img_last , diffusion_results


    def add_scene_gs_dust3r(self, cam_trajs, img_ref, gs_pipe, img_new_num = 3, add_train=True,prompts = None, input_ref =True,diffusion_results=None,full_train=False):
        # generate novel image
        if diffusion_results is None:
            diffusion_results = self.run(cam_trajs, img_ref, gs_pipe.gs, prompts)
        img_new_inds = np.linspace(0, len(diffusion_results) - 1, img_new_num + 1, dtype=int)[1:]
        
        img_news, cam_news, masks =[diffusion_results[0]], [cam_trajs[0]], []
        masks.append(np.ones(diffusion_results[0].shape[:2]))
        for ind in img_new_inds:
            img_new = diffusion_results[ind]
            img_news.append(img_new)
            cam=cam_trajs[ind]
            cam_news.append(cam)
            
            
        img_news=torch.stack(img_news, dim=0)
        new_img=self.dust3r._load_our_images(img_news)
        # new_scene= self.dust3r.run_dust3r_preset(new_img,cam_news)
        new_scene= self.dust3r.run_dust3r(new_img,cam_news)
        scl=self.dust3r.get_scaled(new_scene)
        print('匹配深度图的尺度差距: ',scl)

        pts = self.dust3r.get_pm(new_scene,new_img)
        for i in range(1, img_news.shape[0]):
            cam = cam_news[i]
            _, known_depth, mask, _ = gs_pipe.gs.render(cam)
            known_depth, mask = known_depth.squeeze().detach().cpu().numpy(), mask.squeeze().detach().cpu().numpy()
            mask = (mask < 0.5)
            # we do some dilation to make sure the mask is large enough
            mask = binary_dilation(mask, iterations=25)
            masks.append(mask)
            
            col_img = new_img[i]['img'] * 0.5 + 0.5
            col_img = einops.rearrange(col_img, '1 c h w -> h w c')
            
            estimated_depth = new_scene.get_depthmaps()[i].detach().cpu().numpy()
            Ulog().add_img((estimated_depth - estimated_depth.min()) / (estimated_depth.max() - estimated_depth.min()), "estimated_depth")
            Ulog().add_img(col_img, "col_img")
            
            try:
                new_depth = refine_depth_tmp(known_depth, estimated_depth*scl, mask,rgb_img=col_img)
            except:
                new_depth = estimated_depth * scl
                print("refine depth failed")

            new_pm = depth2pcd_world(new_depth, cam)
            new_pts6d = np.concatenate([new_pm, col_img], axis=-1)
            pcdtmp = PcdMgr(pts3d=new_pts6d[mask])
            Ulog().add_ply(pcdtmp, "pcd_incre_tmp")
            
            # pcdtmp.remove_outliers_near()
            # pcdtmp.remove_outliers()
            pcdtmp.remove_using_traj(cam_trajs)
            new_gs = GaussianMgr().init_from_pts(pcdtmp.pts, mode="fixed", scale = 3e-4)
            gs = gs_pipe.gs.merge(new_gs)
            gs_pipe.gs_update(gs)
            Ulog().add_ply(gs, "pcd_tmp")
            
        if add_train:
            gs_pipe.add_frame(diffusion_results, cam_trajs, masks)
            gs_pipe.train(full_train=False)
            #Ulog().add_ply(gs_pipe.gs, "pcd_after_train")
        
        if input_ref:
            img_last = img_ref
        else:
            img_last = (img_new.cpu().detach().numpy()*255).astype(np.uint8)
            
        return gs_pipe, img_last , diffusion_results



    # def add_scene_mvdust3r(self, cam_trajs, img_ref, gs_pipe, img_new_num = 3, add_train=True,prompts = None, input_ref =True):
    #     # generate novel image
    #     diffusion_results = self.run(cam_trajs, img_ref, gs_pipe.gs, prompts)
    #     img_new = diffusion_results[-1]
        
    #     mp4 = 'cache/diffusion-mvdust3r.mp4'
    #     imageio.mimsave(mp4,(diffusion_results.cpu().numpy()*255.).astype(np.uint8))
        
    #     pts = self.mvdust3r.run(mp4,f_ori=cam_trajs[0].f,min_conf_thr=3.0,n_frame=10)[0]
        
    #     pcdtmp = PcdMgr(pts3d=pts)
    #     # pcdtmp.remove_outliers_near()
    #     new_gs = GaussianMgr().init_from_pts(pcdtmp.pts, mode="knn")
    #     gs = gs_pipe.gs.merge(new_gs)
    #     gs_pipe.gs_update(gs)
    #     gs.save_ply("./cache/pcd_tmp.ply")
        
    #     if add_train:
    #         gs_pipe.add_frame(diffusion_results, cam_trajs)
    #         gs_pipe.train()
    #         gs_pipe.gs.save_ply("./cache/pcd_tmp.ply")
        
    #     if input_ref:
    #         img_last = img_ref
    #     else:
    #         img_last = (img_new.cpu().detach().numpy()*255).astype(np.uint8)
            
    #     return gs_pipe, img_last , diffusion_results