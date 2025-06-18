from omegaconf import OmegaConf
import argparse
import numpy as np
import os
import random
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image

from ops.PcdMgr import PcdMgr
from ops.cam_utils import Mcam, CamPlanner
from ops.ObjMgr import ObjectMgr
from scipy.ndimage import binary_dilation
from ops.utils.depth import refine_depth,depth2pcd_world
from ops.utils.general import seed_everything, save_video
from ops.gs.base import Trainable_Gaussian, GaussianMgr
from ops.utils.logger import Ulog, UlogFilter

def get_real_bg_mask(bg_mask):
    dilate_bg_mask= 1.0 - binary_dilation(1.0 - bg_mask, iterations=10)
    mask=1-dilate_bg_mask
    mask_images = (mask * 255).astype(np.uint8)
    mask_images = Image.fromarray(mask_images).resize((W, H))
    mask_images.save(f"./cache/mask_bg.png")
    mask_images = np.array(mask_images, dtype=np.float32) / 255.0
    mask = (mask_images > 0)
    return mask

def dust3r_pipe(opt):
    from ops.dust3r import Dust3rWrapper
    from ops.PcdMgr import PcdMgr
    global dust3r
    dust3r = Dust3rWrapper(opt.dust3r)
    # preprocess input
    dust3r.load_initial_images([input_image_path],opt)
    global H, W
    H, W = dust3r.images[0]["img"].shape[2:]
    
    background_mask=None
    # generate scene
    dust3r.run_dust3r_init(bg_mask=background_mask)
    bg_pm, pm = dust3r.get_inital_pm()
    cam = dust3r.get_cams()[-1]
    pm=pm.cpu().numpy()
    pm[...,:3] = pm[...,:3] @ cam.getW2C()[:3, :3].T + cam.getW2C()[:3, 3].T
    Mcam.set_default_f(cam.f)
    print("f: ",Mcam.default_f)
    
    objs = ObjectMgr()
    objs.add_pms(pm) 

    pcdtmp = PcdMgr(pts3d=pm.reshape(-1, 6))
    Ulog().add_ply(pcdtmp, "pcd_ori")
    pcdtmp.save_ply(f"{output_dir}/{name}_pcd_ori.ply")
    
    return objs, pcdtmp
    
def seg_pipe(opt):
    from ops.seg_tool import Segment_Tool,Grounded_SAM2_Tool
    segtool = Grounded_SAM2_Tool()
    fgs, bgs, bg_mask = segtool(image,opt.text_fg,opt.text_bg)
    return fgs, bgs, bg_mask
    
def outpaint_pipe(opt):
    if opt.inpaint.outpainting:
        from pipe.lvm_inpaint import Inpaint_Tool
        inpaint_tool= Inpaint_Tool(opt)
        inpaint_tool.outpaint(opt.input_image_path)
        input_image_path = opt.inpaint.outpaint_image_path
    else: 
        inpaint_tool= None
        input_image_path = opt.input_image_path
        
    return input_image_path,inpaint_tool

def inpaint_bg_pipe(opt, inpaint_tool=None):
    if os.path.exists(opt.inpaint.inpaint_bg_path):
        print("load inpaint bg from: ",opt.inpaint.inpaint_bg_path)
        inpaint_bg= Image.open(opt.inpaint.inpaint_bg_path)
    else:
        if inpaint_tool is None:
            from pipe.lvm_inpaint import Inpaint_Tool
            inpaint_tool= Inpaint_Tool(opt)
        inpaint_bg = inpaint_tool.inpaint_bg(image, bg_mask,prompt='empty. nothing. background.'+ opt.text_bg, negative_prompt='entity. object. ' + opt.text_fg)
    return inpaint_bg
    
def replace_obj_once(mask, cropped, ind=0, meshtool=None):
    """
        mask: np.ndarray of shape (H, W)
        cropped: np.ndarray of shape (H, W, 3)
    """
    mask_images = (mask * 255).astype(np.uint8)
    mask_images = Image.fromarray(mask_images).resize((W, H))
    mask_images.save(f"./cache/mask_{ind}.png")
    mask_images = np.array(mask_images, dtype=np.float32) / 255.0
    mask = (mask_images > 0)
    
    img = cropped
    
    pts6d = meshtool(img)
    objpcd = PcdMgr(pts3d=pts6d)
    objpcd.transform_objects()
    # objpcd.save_ply(f"./cache/obj_{ind}.ply")
    # replace object
    objs.add_objects(mask, objpcd.pts)
        
def replace_bg(mask):
    mask=1-mask
    mask_images = (mask * 255).astype(np.uint8)
    mask_images = Image.fromarray(mask_images).resize((W, H))
    mask_images.save(f"./cache/mask_bg.png")
    mask_images = np.array(mask_images, dtype=np.float32) / 255.0
    mask = (mask_images > 0)
    
    bg_img = dust3r._load_images([opt.inpaint.inpaint_bg_path],opt)[0]
    bg_scene= dust3r.run_dust3r(bg_img)
    
    known_depth = dust3r.scene.get_depthmaps()[-1]
    new_depth = refine_depth(known_depth.detach().cpu().numpy(), bg_scene.get_depthmaps()[-1].detach().cpu().numpy(), mask)
    bg_pm = depth2pcd_world(new_depth, Mcam())
    
    col_img = (bg_img[0]['img'] * 0.5 + 0.5).squeeze(0).permute(1,2,0)
    
    bg_pm = np.concatenate([bg_pm, col_img], axis=-1)
    
    bg_pm = bg_pm[mask]
    objs.add_background(bg_pm)

def replace_pipe(opt):
    from pipe.img2pcd import Image2Pcd_Tool
    meshtool = Image2Pcd_Tool(opt)
    if opt.type_3dgen=='instantmesh':
        for obj in fgs:
            print(obj.label)
            replace_obj_once(obj.mask, obj.cropped, obj.idx, meshtool)
    elif opt.type_3dgen=='trellis':
        from ops.seg_tool import crop_to_square
        fg_mask = 1.0 - bg_mask
        masked_fg = image * fg_mask[..., None]
        replace_obj_once(fg_mask, crop_to_square(masked_fg), 0, meshtool)
    
    dilate_bg_mask= 1.0 - binary_dilation(1.0 - bg_mask, iterations=20)
    replace_bg(dilate_bg_mask)

    pcd_new = objs.construct_pcd()
    # pcd_new.remove_outliers()
    Ulog().add_ply(pcd_new, "pcd_final")
    pcd_new.save_ply(f"{output_dir}/{name}_pcd_final.ply")
    print("f: ",Mcam.default_f)
    return pcd_new
    
def get_cam_trajs_list():
    depth = dust3r.depth
    depth_min = depth.min().cpu().item()
    print('depth min:',depth_min)
    mv = 0.98 * abs(depth_min)
    print(mv)
    plan = CamPlanner()
    
    traj_ls = [
            plan.add_traj().move_forward( -1.5* mv, num_frames=48).finish(), 
            plan.add_traj().move_forward(mv, num_frames=24).move_orbit_to(0, 179.999, 0.03*mv, num_frames=24).reinterpolate().finish(),
            plan.add_traj().move_forward(mv, num_frames=24).move_orbit_to(0, -179.999, 0.03*mv, num_frames=24).reinterpolate().finish(),
               ]
    Ulog().add_traj(traj_ls, "traj_list")
    return traj_ls,mv
    
def view_expand_pipe(opt,img_ref, gs_pipe, input_ref=True):
    from pipe.view_extend import Video_Tool
    video_tool = Video_Tool(opt, dust3r)
    
    vid_ref_ls=[]
    for cam_trajs in cam_trajs_ls:
        gs_pipe, img_ref, vid_ref = video_tool.add_scene_gs_dust3r(cam_trajs, img_ref, gs_pipe, input_ref=input_ref,img_new_num=6)
        vid_ref_ls.append(vid_ref)

    return gs_pipe, vid_ref_ls


def img_refine_pipe(opt, gs_pipe,imgnum = 6):
    from ops.img_refiner import Img_refiner
    img_refiner = Img_refiner()
    
    traj_flatten = []
    for traj in cam_trajs_ls:
        for cam in traj:
            traj_flatten.append(cam)
    
    vid = CamPlanner._render_video(gs_pipe.gs, traj_flatten)
    Ulog().add_video(vid, "train_view")

    cam_traj = []
    cam_traj.extend([cam_trajs_ls[1][16],cam_trajs_ls[1][32],cam_trajs_ls[1][-1],cam_trajs_ls[2][16],cam_trajs_ls[2][32]])
        
    refined_imgs = []
    for cam in cam_traj:
        cam.set_size(vid_ref_ls[0][0].shape[0], vid_ref_ls[0][0].shape[1])
        img_render = gs_pipe.gs.render(cam)[0]
        Ulog().add_img(img_render, "img_render")
        img_ref = img_refiner.refine_image(img_render)
        Ulog().add_img(img_ref, "img_ref")
        refined_imgs.append(img_ref)
    
    vid_tensor = np.stack(refined_imgs,axis=0)

    gs_pipe.add_frame(vid_tensor, cam_traj)
    gs_pipe.train(iters=1000,full_train=False,add_ref=1)
    return gs_pipe


def gs_test_pipe(gs):
    # novel view test
    novel_cam_trajs = cam_trajs_ls[1][:-1]+cam_trajs_ls[2][::-1]
    res = []
    for i in range(len(novel_cam_trajs)):
        novel_cam_trajs[i].set_size(vid_ref_ls[0][0].shape[0], vid_ref_ls[0][0].shape[1])
        img = gs.render(novel_cam_trajs[i])[0]
        res.append(img)
    vid_tensor = torch.stack(res,dim=0).detach().cpu()
    Ulog().add_video(vid_tensor, "render_novel_view")
    save_video(vid_tensor, f"{output_dir}/{name}_render_novel_view.mp4")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="room", help='Name of the configuration to load')
    parser.add_argument('--basic_opt', type=str, default='configs/basic.yaml', help='Name of the configuration to load')
    parser.add_argument('--output_dir', type=str, default="./results-360", help='Name of the configuration to load')
    args = parser.parse_args()
    name = args.name
    output_dir = args.output_dir

    os.makedirs('./cache', exist_ok=True)
    Ulog.create(f"main3dgs_{name}", rootdir="./cache")
    Ulog().add_code(__file__)
    Ulog().install_filter(UlogFilter(funcname=None, name="pcd_tmp"))

    # Load the configuration file based on the `name` parameter.
    basic_opt = OmegaConf.load(args.basic_opt)
    opt = OmegaConf.load(f'configs/examples/{name}.yaml')
    opt = OmegaConf.merge(basic_opt, opt) 
    opt.name = name
    opt.replace = False
    if opt.text_fg.strip()+ opt.text_bg.strip() == "":
        opt.replace = False
    
    os.makedirs(output_dir, exist_ok=True) # create results folder
    
    seed_everything(opt)
    input_image_path, inpaint_tool = outpaint_pipe(opt)
    image = Image.open(input_image_path)
    image = np.array(image.convert("RGB"))
    # Image.fromarray(image).save('./cache/image.png')
    
    if opt.replace:
        fgs, bgs, bg_mask= seg_pipe(opt)
        inpaint_bg = inpaint_bg_pipe(opt,inpaint_tool=inpaint_tool)
    
    objs, pcd = dust3r_pipe(opt)
    if opt.replace:
        pcd = replace_pipe(opt)
    from pipe.gs_train import GS_Train_Pipe
    gs_pipe = GS_Train_Pipe(opt, pcd.pts)
    cam_trajs_ls,mv = get_cam_trajs_list()
    # cam_trajs = cam_trajs_ls[0]
    
    gs_pipe, vid_ref_ls = view_expand_pipe(opt,image, gs_pipe, input_ref=True)
    Ulog().add_ply(gs_pipe.gs, "gs_final_norefine")
    # cam from dust3r may be not suitable for gs training
    gs_pipe.gs.save_ply(f"{output_dir}/{name}_gs_final_norefine.ply")
    gs_pipe = img_refine_pipe(opt, gs_pipe,imgnum=opt.refine.imgnum)
    Ulog().add_ply(gs_pipe.gs, "gs_final_2")
    gs_pipe.gs.save_ply(f"{output_dir}/{name}_gs_final_2.ply")
    gs_test_pipe(gs_pipe.gs)
