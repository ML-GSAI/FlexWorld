import os
import json
import torch
import imageio
from ops.cam_utils import Mcam
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from decord import VideoReader

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

class RealEstate10K_Data():
    def __init__(
            self,
            sample_stride=1,
            minimum_sample_stride=1,
            sample_n_frames=49,
            relative_pose=False,
            zero_t_first_frame=False,
            sample_size=[576, 1024],
            rescale_fxy=False,
    ):
        self.root_path = './datasets/RealEstate10K'
        annotation_json='annotation_test.json'
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames

        self.dataset = json.load(open(os.path.join(self.root_path, annotation_json), 'r'))
        self.length = len(self.dataset)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]


    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_video_reader(self, idx):
        video_dict = self.dataset[idx]

        video_path = os.path.join(self.root_path, video_dict['clip_path'])
        video_reader = VideoReader(video_path)
        return video_dict['clip_name'], video_reader, video_dict['caption']

    def load_cameras(self, idx):
        video_dict = self.dataset[idx]
        pose_file = os.path.join(self.root_path, video_dict['pose_file'])
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params

    def get_batch(self, idx):
        clip_name, video_reader, video_caption = self.load_video_reader(idx)
        cam_params = self.load_cameras(idx)
        if not len(cam_params) >= self.sample_n_frames:
            return None, None, None, None
        total_frames = len(cam_params)

        current_sample_stride = self.sample_stride

        # if total_frames < self.sample_n_frames * current_sample_stride:
        #     maximum_sample_stride = int(total_frames // self.sample_n_frames)
        #     current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.sample_n_frames * current_sample_stride
        # start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        start_frame_ind = 0
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        if not end_frame_ind - start_frame_ind >= self.sample_n_frames:
            return None, None, None, None
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        pixel_values = torch.from_numpy(video_reader.get_batch(frame_indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        
        pixel_values = F.interpolate(pixel_values, size=self.sample_size, mode='bilinear', align_corners=False)

        cam_params = [cam_params[indice] for indice in frame_indices]
        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:       # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:                                          # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1],
                                  cam_param.fy * self.sample_size[0],
                                  cam_param.cx * self.sample_size[1],
                                  cam_param.cy * self.sample_size[0]]
                                 for cam_param in cam_params], dtype=np.float32)
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        cam_traj=[]
        for i in range(c2w_poses.shape[0]):
            mcam=Mcam()
            # mcam.set_cam(H=self.sample_size[0],W=self.sample_size[1],f=intrinsics[i,0],c=(intrinsics[i,2],intrinsics[i,3]))
            mcam.set_size(self.sample_size[0],self.sample_size[1])
            mcam.setC2W(c2w_poses[i])
            cam_traj.append(mcam)

        return pixel_values, cam_traj, clip_name, video_caption,frame_indices
    
    def find_idx_by_hash(self, hash):
        for i in range(self.length):
            if self.dataset[i]['clip_name'] == hash:
                return i
        raise ValueError(f'Hash {hash} not found in the dataset')


if __name__ == "__main__":
    import json
    video_dir = "./eval_results/realestate/gt_sampled"

    outcam_dir = "./MotionCtrl/examples/camera_poses_re10"
    outimg_dir = "./MotionCtrl/examples/images_re10"
    
    camctrl_pose_dir= './CameraCtrl/eval_pose_files'
    camctrl_fix_pose_dir= './CameraCtrl/fixed_eval_pose_files'
    os.makedirs(camctrl_fix_pose_dir,exist_ok=True)

    realestate = RealEstate10K_Data(sample_n_frames=49, sample_stride=3,relative_pose=False,zero_t_first_frame=True)
    for fname in os.listdir(video_dir):
        hashval = fname.split('_')[0]
        idx = realestate.find_idx_by_hash(hashval)
        pose_fi=realestate.dataset[idx]['pose_file']
        os.system(f"cp {os.path.join(realestate.root_path,pose_fi)} {os.path.join(camctrl_pose_dir,f'{hashval}.txt')}")
        pixel_values, cam_traj, clip_name, video_caption,frame_indices = realestate.get_batch(idx)
        
        with open(os.path.join(camctrl_fix_pose_dir,f'{hashval}.txt'),'w') as fw:
            with open(os.path.join(camctrl_pose_dir,f'{hashval}.txt')) as f:
                fw.write(f.readline())
                for idx,line in enumerate(f.readlines()):
                    if idx in frame_indices[::2]:
                        fw.write(line)
        c2ws = [cam.getW2C()[:3, :].ravel().tolist() for cam in cam_traj][::3][:14]

        # with open(os.path.join(outcam_dir, f"{hashval}.json"), 'w') as f:
        #     json.dump(c2ws, f)
        
        first_frame = imageio.mimread(os.path.join(video_dir, fname))[0]
        # imageio.imwrite(os.path.join(outimg_dir, f"{hashval}.png"), first_frame)
        print(f"Processed {hashval}")
    