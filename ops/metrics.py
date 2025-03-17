import numpy as np
import math
import os
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import imageio
from ops.utils.fid import calculate_fid_given_paths
from ops.utils.fvd import ScorerFVD,load_videos_given_path
from ops.utils.general import extract_video_to_images
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn.functional as F

class MetricTool():
    def __init__(self):
        self.lpips_model = lpips.LPIPS(net='alex')
        self.ssim_func = StructuralSimilarityIndexMeasure(data_range=255)

    def psnr(self, img1, img2):
        """
            img: [H,W,C] ,[0, 255]
        """
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        mse = np.mean((img1 - img2) ** 2, dtype=np.float64)
        if mse == 0:
            return float('inf')
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    def ssim(self, img1, img2):
        """
            img: [H,W,C] ,[0, 255]
        """
        return ssim(img1, img2, multichannel=True,channel_axis=-1,data_range=255)
    
    def ssim2(self, img1, img2):
        """
            For faster ssim calculation
            img: [H,W,C] ,[0, 255], numpy
        """
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()

        return self.ssim_func(img1, img2).item()

    def lpips(self, img1, img2):
        """
            img: [H,W,C] ,[0, 255]
        """
        img1 = lpips.im2tensor(img1)
        img2 = lpips.im2tensor(img2)
        return self.lpips_model.forward(img1, img2).item()

    def convert_to_np(self, tensor):
        return (tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)

    def transform_and_normalize_rts(self,rts: torch.Tensor):
        """
        Transform the camera coordinate of the estimated poses to be relative to the first frame,
        and normalize the translation scale using the furthest frame.

        Args:
            rts (torch.Tensor): The camera poses, shape (N, 4, 4).

        Returns:
            torch.Tensor: Transformed and normalized camera poses, shape (N, 4, 4).
        """
        # Ensure the input is a tensor
        if not isinstance(rts, torch.Tensor):
            rts = torch.tensor(rts)

        # Get the first frame's RT
        first_rt = rts[0]

        # Transform all RTs to be relative to the first frame
        relative_rts = torch.matmul(torch.inverse(first_rt), rts)

        # Calculate the translation distances
        translations = relative_rts[:, :3, 3]
        distances = torch.norm(translations, dim=1)

        # Use the furthest frame's distance to normalize
        max_distance = torch.max(distances)
        normalized_rts = relative_rts.clone()
        normalized_rts[:, :3, 3] /= max_distance

        return normalized_rts

    def calculate_cam_err(self, pred_rts: torch.Tensor, gt_rts: torch.Tensor):
        """
        Calculate the camera pose error between the predicted and ground truth camera poses.

        Args:
            pred_rts (torch.Tensor): The predicted camera poses, shape (N, 4, 4).
            gt_rts (torch.Tensor): The ground truth camera poses, shape (N, 4, 4).

        Returns:
            torch.Tensor: The camera pose errors, shape (N,).
        """
        # Ensure the input is a tensor
        if not isinstance(pred_rts, torch.Tensor):
            pred_rts = torch.tensor(pred_rts)
        if not isinstance(gt_rts, torch.Tensor):
            gt_rts = torch.tensor(gt_rts)

        # Normalize the camera poses
        pred_rts = self.transform_and_normalize_rts(pred_rts)
        gt_rts = self.transform_and_normalize_rts(gt_rts)

        # Calculate the camera pose errors
        rot_errs = torch.zeros(pred_rts.shape[0])
        trans_errs = torch.zeros(pred_rts.shape[0])
        for i in range(pred_rts.shape[0]):
            pred_rt = pred_rts[i]
            gt_rt = gt_rts[i]

            # Calculate the rotation error
            pred_rot = pred_rt[:3, :3]
            gt_rot = gt_rt[:3, :3]
            rot_err = torch.acos(((torch.trace(torch.matmul(pred_rot, gt_rot.T)) - 1) / 2).clamp(max=1,min=-1))
            if rot_err.isnan():
                print(pred_rot, gt_rot)
                print(torch.matmul(pred_rot, gt_rot.T))
                print(torch.trace(torch.matmul(pred_rot, gt_rot.T)))


            # Calculate the translation error
            pred_trans = pred_rt[:3, 3]
            gt_trans = gt_rt[:3, 3]
            trans_err = torch.norm(pred_trans - gt_trans)
            
            rot_errs[i] = rot_err
            trans_errs[i] = trans_err
        
        final_rot_err = torch.mean(rot_errs)
        final_trans_err = torch.mean(trans_errs)
        
        return final_rot_err, final_trans_err
    

    def calculate_cam_tmp(self, pred_rts: torch.Tensor, gt_rts: torch.Tensor):
        """
        Calculate the camera pose error between the predicted and ground truth camera poses.

        Args:
            pred_rts (torch.Tensor): The predicted camera poses, shape (N, 4, 4).
            gt_rts (torch.Tensor): The ground truth camera poses, shape (N, 4, 4).

        Returns:
            torch.Tensor: The camera pose errors, shape (N,).
        """
        # Ensure the input is a tensor
        if not isinstance(pred_rts, torch.Tensor):
            pred_rts = torch.tensor(pred_rts)
        if not isinstance(gt_rts, torch.Tensor):
            gt_rts = torch.tensor(gt_rts)

        # Normalize the camera poses

        # Calculate the camera pose errors
        rot_errs = torch.zeros(pred_rts.shape[0])
        trans_errs = torch.zeros(pred_rts.shape[0])
        for i in range(pred_rts.shape[0]):
            pred_rt = pred_rts[i]
            gt_rt = gt_rts[i]

            # Calculate the rotation error
            pred_rot = pred_rt[:3, :3]
            gt_rot = gt_rt[:3, :3]
            rot_err = torch.acos(((torch.trace(torch.matmul(pred_rot, gt_rot.T)) - 1) / 2).clamp(max=1,min=-1))
            if rot_err.isnan():
                print(pred_rot, gt_rot)
                print(torch.matmul(pred_rot, gt_rot.T))
                print(torch.trace(torch.matmul(pred_rot, gt_rot.T)))


            # Calculate the translation error
            pred_trans = pred_rt[:3, 3]
            gt_trans = gt_rt[:3, 3]
            trans_err = torch.norm(pred_trans - gt_trans)
            
            rot_errs[i] = rot_err
            trans_errs[i] = trans_err
        
        final_rot_err = torch.mean(rot_errs)
        final_trans_err = torch.mean(trans_errs)
        
        return final_rot_err, final_trans_err

    def calculate_visual_metrics(self, video1_path, video2_path):
        """
            FID and FVD
            video1_path: str, dir of video1
            video2_path: str, dir of video2
        """
        img_cache_dir1 = "cache/video1_frames"
        img_cache_dir2 = "cache/video2_frames"
        os.system(f"rm -rf {img_cache_dir1}")   # remove cache
        os.system(f"rm -rf {img_cache_dir2}")   # remove cache
        video1_path_ls = sorted([os.path.join(video1_path, f) for f in os.listdir(video1_path) if f.endswith('.mp4')])
        video2_path_ls = sorted([os.path.join(video2_path, f) for f in os.listdir(video2_path) if f.endswith('.mp4')])
        for p in video1_path_ls:
            extract_video_to_images(p, img_cache_dir1)
        for p in video2_path_ls:
            extract_video_to_images(p, img_cache_dir2)
        # FID
        fid = calculate_fid_given_paths([img_cache_dir1, img_cache_dir2])
        
        # FVD
        fvd_scorer = ScorerFVD(device="cuda",pretrained_path='./ops/utils/fvd_tool/i3d_pretrained_400.pt')
        v1=load_videos_given_path(video1_path,resolution=256)
        v2=load_videos_given_path(video2_path,resolution=256)
        fvd = fvd_scorer.fvd(v1, v2)
        
        return {
            'fid': fid,
            'fvd': fvd
        }
        

    def calculate_sim_metrics(self, video1: torch.Tensor, video2: torch.Tensor,skip_first_frame=True):
        """
            video1: [N,C,H,W], [0, 1]
            video2: [N,C,H,W], [0, 1]
        """
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        if skip_first_frame:
            video1 = video1[1:]
            video2 = video2[1:]
            
        if video1.shape!=video2.shape:
            print(f"Different shape: {video1.shape} vs {video2.shape}")
            video1 = F.interpolate(video1, size=(video2.shape[2], video2.shape[3]), mode='bilinear', align_corners=False)

        for frame1, frame2 in zip(video1, video2):
            frame1, frame2 = self.convert_to_np(frame1), self.convert_to_np(frame2)
            psnr_values.append(self.psnr(frame1, frame2))
            # ssim_values.append(self.ssim(frame1, frame2))
            ssim_values.append(self.ssim2(frame1, frame2))
            # print(self.ssim2(frame1, frame2)-self.ssim(frame1, frame2))
            lpips_values.append(self.lpips(frame1, frame2))

        return {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values),
            'lpips': np.mean(lpips_values)
        }
        
    def calculate_sim_metrics_from_videos_list(self, video1_ls, video2_ls):
        """
            PSNR, SSIM, LPIPS
            video1_ls: str, list of video1_path
            video2_ls: str, list of video2_path
        """
        def read_video(video_path):
            try:
                video = imageio.mimread(video_path)
                video = np.stack(video)
                video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
                return video
            except Exception as e:
                print(f"Error reading video {video_path}: {e}")
                raise e
            
        video_num = len(video1_ls)
        assert video_num == len(video2_ls)
        psnr, ssim, lpips = 0, 0, 0
        with tqdm(list(zip(video1_ls,video2_ls)), desc="Processing videos") as pbar:
            for video1_path,video2_path in pbar:
                try:
                    video1=read_video(video1_path)
                    video2=read_video(video2_path)
                    res = self.calculate_sim_metrics(video1, video2)
                    psnr += res['psnr']
                    ssim += res['ssim']
                    lpips += res['lpips']
                except:
                    print(f"Error processing video {video1_path} and {video2_path}")
                    video_num -= 1
                    continue
                
                
        psnr /= video_num
        ssim /= video_num
        lpips /= video_num
        return {
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips
        }
        