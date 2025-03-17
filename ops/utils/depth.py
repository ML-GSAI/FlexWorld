import numpy as np
import cv2
from scipy.ndimage import binary_dilation
from PIL import Image
import torch

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    '''
    Copied from Marigold
    '''
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

def refine_depth2(render_dpt: np.ndarray, ipaint_dpt: np.ndarray, ipaint_msk: np.ndarray, iters=100, blur_size=15, scaled=True):
    """
        refine depth map.
        Args:
            render_dpt: rendered depth map, [H, W]
            ipaint_dpt: ipainted depth map, [H, W]
            ipaint_msk: ipainted mask
            scaled: whether to scale ipaint_dpt to the same range as render_dpt
    """
    if scaled:
        ipaint_dpt, scale, shift = align_depth_least_square(render_dpt, ipaint_dpt, ~ipaint_msk)
        print(f"refine_depth: found scaled value shift {shift} and scale {scale}")
    keep_render_dpt = render_dpt[~ipaint_msk]
    keep_ipaint_dpt = ipaint_dpt[~ipaint_msk]


    keep_adjust_dpt = keep_render_dpt - keep_ipaint_dpt
    # iterative refinement
    complete_adjust = np.zeros_like(ipaint_dpt)
    for i in range(iters):
        complete_adjust[~ipaint_msk] = keep_adjust_dpt
        complete_adjust = cv2.blur(complete_adjust,(blur_size,blur_size))
    # complete_adjust[~ipaint_msk] = keep_adjust_dpt
    ipaint_dpt = ipaint_dpt + complete_adjust        
    return ipaint_dpt

def refine_depth(render_dpt: np.ndarray, ipaint_dpt: np.ndarray, ipaint_msk: np.ndarray, iters=100, blur_size=15, scaled=True):
    """
        refine depth map.
        Args:
            render_dpt: rendered depth map, [H, W]
            ipaint_dpt: ipainted depth map, [H, W]
            ipaint_msk: ipainted mask
            scaled: whether to scale ipaint_dpt to the same range as render_dpt
    """
    keep_render_dpt = render_dpt[~ipaint_msk]
    keep_ipaint_dpt = ipaint_dpt[~ipaint_msk]
    if scaled:
        # render_min, render_max = keep_render_dpt.min(), keep_render_dpt.max()
        # ipaint_min, ipaint_max = keep_ipaint_dpt.min(), keep_ipaint_dpt.max()
        render_min, render_max = np.percentile(keep_render_dpt, 10), np.percentile(keep_render_dpt, 90)
        ipaint_min, ipaint_max = np.percentile(keep_ipaint_dpt, 10), np.percentile(keep_ipaint_dpt, 90)
        print(render_min, render_max, ipaint_min, ipaint_max)

        ipaint_dpt = (ipaint_dpt - ipaint_min) / (ipaint_max - ipaint_min) * (render_max - render_min) + render_min
        keep_ipaint_dpt = ipaint_dpt[~ipaint_msk]

    keep_adjust_dpt = keep_render_dpt - keep_ipaint_dpt
    # iterative refinement
    complete_adjust = np.zeros_like(ipaint_dpt)
    for i in range(iters):
        complete_adjust[~ipaint_msk] = keep_adjust_dpt
        complete_adjust = cv2.blur(complete_adjust,(blur_size,blur_size))
    # complete_adjust[~ipaint_msk] = keep_adjust_dpt
    ipaint_dpt = ipaint_dpt + complete_adjust        
    return ipaint_dpt

def guided_filter(I, p, r, eps):
    """
    Guided Filter

    Args:
        I: guide image (彩色图或者灰度图), shape: (H, W, C) or (H, W)
        p: filtering input image (深度图), shape: (H, W)
        r: radius of the guided filter
        eps: regularization parameter
    """
    if len(I.shape) == 3:
        I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    else:
        I_gray = I
    I_gray = I_gray.astype(np.float64)
    p = p.astype(np.float64)

    # 计算均值
    mean_I = cv2.blur(I_gray, (r, r))
    mean_p = cv2.blur(p, (r, r))

    # 计算协方差和方差
    mean_Ip = cv2.blur(I_gray * p, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.blur(I_gray * I_gray, (r, r))
    var_I = mean_II - mean_I * mean_I

    # 计算 a 和 b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 计算均值
    mean_a = cv2.blur(a, (r, r))
    mean_b = cv2.blur(b, (r, r))

    # 计算输出
    q = mean_a * I_gray + mean_b
    return q.astype(np.float32)



def refine_depth_tmp(render_dpt: np.ndarray, ipaint_dpt: np.ndarray, ipaint_msk: np.ndarray,rgb_img=None,scaled=True):
    """
        refine depth map.
        Args:
            render_dpt: rendered depth map, [H, W]
            ipaint_dpt: ipainted depth map, [H, W]
            ipaint_msk: ipainted mask
            scaled: whether to scale ipaint_dpt to the same range as render_dpt
    """
    keep_render_dpt = render_dpt[~ipaint_msk]
    keep_ipaint_dpt = ipaint_dpt[~ipaint_msk]
    if scaled:
        # render_min, render_max = keep_render_dpt.min(), keep_render_dpt.max()
        # ipaint_min, ipaint_max = keep_ipaint_dpt.min(), keep_ipaint_dpt.max()
        try:
            render_min, render_max = np.percentile(keep_render_dpt, 10), np.percentile(keep_render_dpt, 90)
            ipaint_min, ipaint_max = np.percentile(keep_ipaint_dpt, 10), np.percentile(keep_ipaint_dpt, 90)
            print(render_min, render_max, ipaint_min, ipaint_max)

            ipaint_dpt = (ipaint_dpt - ipaint_min) / (ipaint_max - ipaint_min) * (render_max - render_min) + render_min
            keep_ipaint_dpt = ipaint_dpt[~ipaint_msk]
        except:
            print("Error in scaling depth")
            

    render_dpt_filled = render_dpt.copy()
    render_dpt_filled[ipaint_msk] = ipaint_dpt[ipaint_msk]

    rgb_img = ((rgb_img * 255).cpu().numpy()).astype(np.uint8)
    # 应用引导滤波
    refined_depth = guided_filter(rgb_img, render_dpt_filled, 15, 1e-8)  
    return refined_depth

def depth2pcd(depth, cam):
    """
        depth: [H, W]
        cam: Mcam
        return pcd in cam openCV space, [H, W, 3]
    """
    H, W = depth.shape
    f = cam.f
    cx, cy = cam.c
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = (x - cx) / f
    y = (y - cy) / f
    z = np.array(depth)
    pts = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1)
    return pts

def depth2pcd_world(depth, cam):
    """
        depth: [H, W], should be positive
        cam: Mcam
        return pcd in openGL world space, [H, W, 3]
    """
    pts = depth2pcd(depth, cam)
    pts[:, :, [2,1]] = -pts[:, :, [2, 1]]
    pts = pts @ cam.getC2W()[:3, :3].T + cam.getC2W()[:3, 3].T
    return pts

def depth2pcd_world_torch(depth, cam):
    """
        pcd: [N, 3], in openGL world space
        cam: Mcam
        return depth in openGL world space, [H, W]
    """
    if isinstance(depth, torch.Tensor):
        pass
    elif isinstance(depth, np.ndarray):
        depth = torch.tensor(depth, device="cuda")
    else:
        raise ValueError("Unsupported type of depth, {}".format(type(depth)))
    device = depth.device
    H, W = depth.shape
    f = torch.tensor(cam.f, device=device)
    cx, cy = torch.tensor(cam.c, device=device)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    x = (x - cx) / f
    y = (y - cy) / f
    z = depth
    pts = torch.stack((torch.multiply(x, z), torch.multiply(y, z), z), axis=-1)
    pts[:, :, [2,1]] = -pts[:, :, [2, 1]]
    pts = pts @ torch.tensor(cam.getC2W()[:3, :3].T, device=device) + torch.tensor(cam.getC2W()[:3, 3].T, device=device)
    return pts

def visualize_depth(predicted_depth,out_path=None):
    '''
    predicted_depth: [H, W], range in any
    '''
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    import torch
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    elif isinstance(depth, np.ndarray):
        depth_np = depth
    else:
        raise ValueError("Unsupported type of image, {}".format(type(depth)))
    depth_save = Image.fromarray((depth_np * 255).astype("uint8"))
    if out_path is not None:
        depth_save.save(out_path)
    return depth_np