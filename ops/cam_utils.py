import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from dataclasses import dataclass
import torch
from tqdm import tqdm
import copy
import math


def create_position_interpolator(positions, times):
    """创建位置插值器，使用三次样条"""
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(times, positions)
    return lambda t: cs(t)

def create_rotation_interpolator(rotations, times):
    """创建旋转插值器，使用球面线性插值"""
    from scipy.spatial.transform import Slerp
    slerp = Slerp(times, rotations)
    return lambda t: slerp([t])[0]

class AutoPlanner():
    def __init__(self, pcd, basecam):
        from ops.gs.base import GaussianMgr
        if isinstance(pcd, GaussianMgr):
            pcd.pts = pcd.get_pts6d()
        self.gs = GaussianMgr().init_from_pts(pcd.pts, mode="fixed", scale=0.001)
        self.basecam = basecam

    def find_nbv(self, src_cam):
        max_try = 1000
        for _ in range(max_try):
            dxdy = np.random.random(size=2) * 2 - 1
            dxdy = dxdy * 0.3 # -0.3 - +0.3
            pos = src_cam.T.copy()
            pos[:2] += dxdy
            azi = np.random.random() * 360
            newcam = src_cam.copy()
            newcam.T = pos
            newcam.set_orbit_inplace(0, azi)
            
    def wander(self, startcam=None, rec=[-1, 1], numframes=50):
        # 随机游走，rec是游走范围，numframes是游走帧数
        if startcam is None:
            startcam = self.basecam
        xyz = np.random.random(rec, size=(numframes, 3))
        return xyz
    
        



    def is_collision_single_cam(self, cam, radii_threshold=30, num_threshold=2):
        # 我们使用固定半径初始化高斯，如果一定数量的高斯的半径大于阈值，我们认为发生了碰撞
        info = self.gs.render(cam)[-1]
        if (info["radii"] > radii_threshold).sum() > num_threshold:
            return True
        return False
    
    def is_collision(self, traj):
        for i in range(len(traj)):
            if self.is_collision_single_cam(traj[i]):
                return True
        return False
    
    def _get_first_collision_idx(self, traj):
        for i in range(len(traj)):
            if self.is_collision_single_cam(traj[i]):
                return i
        return -1
    
    def _search_simple_orbit(self, camreal, radius):
        # push in and push out the cam, until there is no collision
        max_try = 1000
        cam = camreal.copy()
        central_pt = cam.T - radius * cam.R[:,2]
        def print_r(cam):
            print("radius", np.linalg.norm(cam.T - central_pt))
        trajs = CamPlanner().add_traj(startcam=cam).move_forward(radius, num_frames = 100).drop()
        for idx, cam in enumerate(trajs):
            if not self.is_collision_single_cam(cam):
                print(f"Found a good radius when pushing in {idx}")
                return cam
        trajs = CamPlanner().add_traj(startcam=camreal).move_forward(-radius, num_frames = 100).drop()
        for idx, cam in enumerate(trajs):
            if not self.is_collision_single_cam(cam):
                print(f"Found a good radius when pushing out {idx}")
                return cam
        print("Unable to find a good position in 2*orbit, moving out to more")
        for _ in range(max_try):
            trajs = CamPlanner().add_traj(startcam=trajs[-1]).move_forward(-0.02).drop()
            print_r(trajs[-1])
            if not self.is_collision_single_cam(trajs[-1]):
                print(f"Found a good radius when pushing out {idx}")
                return trajs[-1]
            
    def correct_orbit_once(self, traj, central_pt, call_stack):
        while self.is_collision(traj):
            first_idx = self._get_first_collision_idx(traj)
            print(f"Collision detected at idx {first_idx}, searching for a good position")
            radius = traj[first_idx].get_orbit(central_pt)[2]
            newcam = self._search_simple_orbit(traj[first_idx], radius)
            new_radius = np.linalg.norm(newcam.T - central_pt)
            print(f"New radius found: {new_radius}")
            traj[first_idx] = newcam

            basecam_new = self.basecam.copy()
            ele, azi, radius = basecam_new.get_orbit(central_pt)
            basecam_new.set_orbit(ele, azi, new_radius, central_pt)
            tar_ele, tar_azi, _, d_radius, num_frames = call_stack
            traj_new = CamPlanner().add_traj(startcam=basecam_new).move_orbit_to(tar_ele, tar_azi, new_radius, d_radius, num_frames).finish()
            traj = traj[:first_idx] + traj_new[first_idx:]
        return traj

        
    def move_orbit_to(self, tar_ele, tar_azi, radius, d_radius=0, num_frames=30):
        # only support fixed radius now
        if d_radius != 0:
            raise ValueError("Only support fixed radius now")
        traj = CamPlanner().add_traj(startcam=self.basecam).move_orbit_to(tar_ele, tar_azi, radius, d_radius, num_frames).finish()
        self.visualize_traj(traj, "cache/traj0.png")
        basecam = self.basecam.copy()
        callstack = (tar_ele, tar_azi, radius, d_radius, num_frames)
        central_pt = CamPlanner().add_traj(startcam=basecam).move_forward(radius).finish()[-1].T
        k = 0
        while self.is_collision(traj):
            traj = self.correct_orbit_once(traj, central_pt, callstack)
            k += 1
            self.visualize_traj(traj, f"cache/traj{k}.png")
            trajxy = np.array([cam.T[[0,2]] for cam in traj])
            if k > 5:
                return traj
            newxy = self.smooth_traj(trajxy)
            for i in range(len(trajxy)):
                traj[i].T[[0,2]] = newxy[i]
            k += 1
            self.visualize_traj(traj, f"cache/traj{k}.png")


        return traj


    def move_with_length(self, left, up, forward, num_frames=30):
        backstep = 2

        traj = CamPlanner().add_traj(startcam=self.basecam).move_with_length(left, up, forward, num_frames).finish()
        basecam = self.basecam.copy()
        for i in range(len(traj)):
            cam = traj[i]
            if self.is_collision_single_cam(cam):
                print(f"Collision detected at idx {i}, go back for {backstep} and re-step")
                target_cam = cam[max(0, i - backstep)]
                newtraj = CamPlanner().add_traj(startcam=basecam).move_to(target_cam.T, num_frames=num_frames).finish()
                return newtraj
        print("No collision detected")
        return newtraj

    def visualize_traj(self, traj, output_path="cache/traj.png"):
        trajxy = [cam.T[[0,2]] for cam in traj]
        from PIL import Image, ImageDraw
        trajxy = np.array(trajxy)
        center = np.array([0, 0])
        size = trajxy.max() - trajxy.min()
        trajxy = (trajxy - center) * 600 + 256
        #trajxy = [trajxy[i] + [256, 256] for i in range(len(trajxy))]
        img = Image.new("RGB", (512, 512), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for i in range(len(trajxy)):
            draw.ellipse([trajxy[i][0] - 3, trajxy[i][1] - 3, trajxy[i][0] + 3, trajxy[i][1] + 3], fill='red')

        img.save(output_path)



    def smooth_traj(self, traj, window_length=31, polyorder=2):
        from scipy.signal import savgol_filter
        """
        对一系列xy坐标轨迹进行平滑
        :param traj: 轨迹点列表每个点是一个包含x和y的元组
        :param window_length: 滑动窗口的长度，必须是奇数
        :param polyorder: 多项式阶数
        :return: 平滑后的轨迹点列表
        """
        traj = np.array(traj)
        x = traj[:, 0]
        y = traj[:, 1]

        x_smooth = savgol_filter(x, window_length, polyorder)
        y_smooth = savgol_filter(y, window_length, polyorder)

        return list(zip(x_smooth, y_smooth))

    @staticmethod
    def smooth_trajectory2(traj, s=0.1):
        from scipy.interpolate import splprep, splev
        """
        使用B样条对一系列xy坐标轨迹进行平滑
        :param traj: 轨迹点列表，每个点是一个包含x和y的元组
        :param s: 平滑因子，s越大，曲线越平滑
        :return: 平滑后的轨迹点列表
        """
        traj = np.array(traj)
        x = traj[:, 0]
        y = traj[:, 1]

        # 使用B样条进行平滑
        tck, u = splprep([x, y], s=s)
        unew = np.linspace(0, 1.0, num=len(x) * 10)
        out = splev(unew, tck)

        x_smooth = out[0]
        y_smooth = out[1]

        return list(zip(x_smooth, y_smooth))
    




class TrajMove():
    def __init__(self):
        pass


    @staticmethod
    def interupt_traj(traj, eps=0.01, sep=3, smoothtime = 13):
        if len(traj) == 0:
            return
        elem = traj[0]
        trajraw = traj
        traj = copy.deepcopy(traj)
        if isinstance(elem, Mcam):
            pass
        elif isinstance(elem, list):
            record_len = []
            for i in range(len(elem)):
                record_len.append(len(elem[i]))
            traj = [traji[i]  for traji in traj for i in range(len(traji))]


        sinadd = np.sin(np.linspace(0, 2 * np.pi * 20 , len(traj)))
        for i, cam in enumerate(traj):
            cam.T += np.array([0, sinadd[i] * eps, 0])


        # for cam in traj[::sep]:
        #     randxyz = np.random.random(3) * eps
        #     #randxyz[2] = 0
        #     cam.T += randxyz

        if isinstance(elem, list):
            traj = [[traj[i] for i in range(record_len[j])] for j in range(len(elem))]
        print(trajraw[0])
        for _ in range(smoothtime):
            traj[0] = trajraw[0].copy()
            traj = TrajMove.smooth_traj(traj)
        traj[0] = trajraw[0]
        print(traj[0])
        return traj

    @staticmethod
    def smooth_traj(traj, window_length=31, polyorder=2):
        from scipy.signal import savgol_filter
        """
        对一系列xy坐标轨迹进行平滑
        :param traj: 轨迹点列表每个点是一个包含x和y的元组
        :param window_length: 滑动窗口的长度，必须是奇数
        :param polyorder: 多项式阶数
        :return: 平滑后的轨迹点列表
        """
        trajxy = np.array([cam.T[[0,1]] for cam in traj])
        x = trajxy[:, 0]
        y = trajxy[:, 1]

        x_smooth = savgol_filter(x, window_length, polyorder)
        y_smooth = savgol_filter(y, window_length, polyorder)

        for i, cam in enumerate(traj):
            cam.T[[0,1]] = x_smooth[i], y_smooth[i]
        return traj

    

    
class RotationModel():
    def __init__(self, max_angle=360.0, t_total=5.0, t_middle=3.0) -> None:
        self.s = max_angle
        self.t_total = t_total
        self.t_middle = t_middle
        self.t_begin = (t_total - t_middle) / 2.0

        self.a = (self.s) / (self.t_begin ** 2 + self.t_begin * t_middle)
    
    def degfromT(self, t):
        v = self.a * self.t_begin
        if t < self.t_begin:
            return self.a * t * t / 2
        elif t < self.t_begin + self.t_middle:
            return v * self.t_begin / 2 + v * (t - self.t_begin)
        else:
            return self.s - self.a * ((self.t_total - t) ** 2) / 2



class CamPlanner():
    """ Store camera trajectory and render videos
    """

    suppress_tqdm = False
    def __init__(self, startcam=None):
        self.camtrajs = {}
    

    def add_traj(self, startcam=None, name=None):
        if getattr(self, "startcam", None) is not None:
            raise ValueError("Cam planning should finish() before adding new trajectory")
        if startcam is None:
            startcam = Mcam()
        if name is None:
            name = str(len(self.camtrajs))
        self.startcam = startcam
        self.curcam = startcam.copy()
        self.current_traj = [self.curcam]
        if name in self.camtrajs:
            raise ValueError(f"Trajectory {name} already exists")
        self.camtrajs[name] = self.current_traj
        self.curcam = startcam.copy()
        self.curname = name
        return self

    def get_all_traj(self):
        return list(self.camtrajs.values())
    
    def get_all_traj(self):
        return list(self.camtrajs.values())
    
    # simple move functions, wrap of move_with_rotate
    def move_forward(self, length, num_frames=30):
        '''length: scalar +/-'''
        self.move_with_rotate(-length, 0, direction="z", num_frames=num_frames)
        return self
    
    def move_left(self, length, num_frames=30):
        '''length: scalar +/-'''
        self.move_with_rotate(-length, 0, direction="x", num_frames=num_frames)
        return self
    
    def move_up(self, length, num_frames=30):
        '''length: scalar +/-'''
        self.move_with_rotate(length, 0, direction="y", num_frames=num_frames)
        return self
    
    def rotate_left(self, angle, num_frames=30):
        '''angle: deg'''
        self.move_with_rotate(0, angle, rotate="left", num_frames=num_frames)
        return self
    
    def rotate_up(self, angle, num_frames=30):
        '''angle: deg'''
        self.move_with_rotate(0, angle, rotate="up", num_frames=num_frames)
        return self

    def move_to(self, target, num_frames=30):
        '''target: [3]'''
        self.move_with_rotate(0, 0, target=target, num_frames=num_frames)
        return self

    def move_with_length(self, left, up, forward, num_frames=30):
        ''' x, y, z: scalar +/-, the strength to go in the direction of x, y, z
        '''
        target = self.curcam.T - left * self.curcam.R[:,0] + up * self.curcam.R[:,1] - forward * self.curcam.R[:,2]
        self.move_to(target, num_frames)
        return self


    def _parse_rotate(self, rotate):
        if rotate == "up":
            return lambda angle: R.from_euler("xyz", [angle, 0, 0], degrees=True)
        elif rotate == "down":
            return lambda angle: R.from_euler("xyz", [-angle, 0, 0], degrees=True)
        elif rotate == "left":
            return lambda angle: R.from_euler("xyz", [0, angle, 0], degrees=True)
        elif rotate == "right":
            return lambda angle: R.from_euler("xyz", [0, angle, 0], degrees=True)
        else:
            raise ValueError("Invalid rotate description, only support up/down/left/right")
            
    def _parse_direction(self, direction):
        ''' return axis index'''
        direction = direction.lower()
        if direction == "z":
            return 2
        elif direction == "x":
            return 0
        elif direction == "y":
            return 1
        else:
            raise ValueError("Invalid direction description, only support z/x/y")
        
    
    def move_headbanging_circle(self, maxdeg, num_frames=30, round=1, fullround=1):
        # copied from LucidDreamer
        
        nviews_per_round = 30 // round
        radius = np.concatenate((np.linspace(0, maxdeg, nviews_per_round*round), maxdeg*np.ones(nviews_per_round*fullround), np.linspace(maxdeg, 0, nviews_per_round*round)))
        radius = radius[:num_frames]
        thlist  = 2.66*radius * np.sin(np.linspace(0, 2*np.pi*(round+fullround+round), num_frames))
        philist = radius * np.cos(np.linspace(0, 2*np.pi*(round+fullround+round), num_frames))
        assert len(thlist) == len(philist)

        render_poses = np.zeros((len(thlist), 3, 4))
            

        for i in range(num_frames):
            #self.curcam.T = self.curcam.T + render_poses[i,:3,3]
            #self.curcam.R = np.matmul(render_poses[i,:3,:3], self.curcam.R )
            self.curcam = self.curcam.set_orbit_inplace(thlist[i], philist[i])
            self.current_traj.append(self.curcam)
            self.curcam = self.curcam.copy()

        return self

    def move_with_rotate(self, move_len, angle, target=None, rotate="up", direction="z", num_frames=30):
        '''rotate: up/down/left/right, direction: z/x/y'''
        rotate_func = self._parse_rotate(rotate)
        rotate_mat = rotate_func(angle).as_matrix()
        dir_ind = self._parse_direction(direction)
        curpose = self.curcam.getC2W()
        curpose[:3, 3] += move_len * curpose[:3, dir_ind]
        curpose[:3, :3] = np.dot(curpose[:3, :3], rotate_mat)
        if target is not None:
            curpose[:3, 3] = target
        self.move_pose_to(curpose, num_frames)
        return self

    def move_pose_to(self, targetRT, num_frames=30):
        ''' targetRT: [4,4] R uses slerp, T uses linear interpolation
        '''
        # interpolate R
        targetR = targetRT[:3,:3]
        targetT = targetRT[:3,3]
        curT = self.curcam.T
        stepT = (targetT - curT) / num_frames

        curR = self.curcam.R
        rot1 = R.from_matrix(curR)
        rot2 = R.from_matrix(targetR)

        times = [0, num_frames]
        key_rots = R.from_quat([rot1.as_quat(), rot2.as_quat()])

        slerp = Slerp(times, key_rots)

        for i in range(1, num_frames + 1):
            self.curcam.T += stepT
            curR = slerp(i).as_matrix()
            self.curcam.R = curR
            self.current_traj.append(self.curcam)
            self.curcam = self.curcam.copy()
        
        return self
    
    def extend(self, scale=2):
        new_current_traj = self.extend_traj(self.current_traj, scale)
        self.current_traj = new_current_traj
        return self
    
    def extend_traj(self, traj, scale=2):
        new_current_traj = []
        for i in range(len(traj)-1):
            sta= traj[i]
            end = traj[(i+1)]
            trajnew = CamPlanner().add_traj(startcam=sta).move_pose_to(end.getC2W(), num_frames=scale).finish()[:-1]
            new_current_traj.extend(trajnew)
        new_current_traj.append(traj[-1])
        return new_current_traj
    

    def move_interpolate(self, intlist, num_frames=None):
        """
        Create a camera trajectory using smooth interpolation based on given path points.
        
        Parameters:
        intlist: [(left, forward, azi, len)] list, where each element contains:
            - left: x-axis offset
            - forward: z-axis offset
            - azi: azimuth angle (degrees)
            - len: number of frames for this segment
        num_frames: Optional, total number of frames. If set, it will override the sum of segment lengths.
        
        Returns:
        self: For chained calls.
        """
        sumlen = 0
        for tup in intlist:
            sumlen += tup[3]
        
        if num_frames is None:
            num_frames = int(sumlen)
        
        cameras = []
        keyframe_positions = []
        
        cameras.append(self.curcam.copy())
        
        # Calculate the target camera and keyframe positions for each point
        for i, tup in enumerate(intlist):
            assert len(tup) == 4
            x, z, azi, length = tup
            
            targetcam = self.curcam.copy()
            targetcam.T = np.array([-x, 0, -z])
            targetcam.set_orbit_inplace(0, azi)
            
            cameras.append(targetcam)
            keyframe_positions.append(int(sum(t[3] for t in intlist[:i+1]) / sumlen * num_frames))
        
        keyframe_positions[-1] = num_frames
        
        c2ws = [cam.getC2W() for cam in cameras]
        
        positions = np.array([c2w[:3, 3] for c2w in c2ws])
        
        from scipy.spatial.transform import Rotation as R, Slerp
        
        rotations = R.from_matrix([c2w[:3, :3] for c2w in c2ws])
        
        times = np.array([0] + keyframe_positions) / num_frames
        
        position_interp = create_position_interpolator(positions, times)
        
        slerp = Slerp(times, rotations)
        
        for i in range(num_frames):
            t = i / num_frames
            
            pos = position_interp(t)
            rot_matrix = slerp([t])[0].as_matrix()
            
            newcam = self.curcam.copy()
            newcam.T = pos
            newcam.R = rot_matrix
            
            self.current_traj.append(newcam)
            self.curcam = newcam.copy()
        
        return self


        
    def move_orbit_to(self, tar_ele, tar_azi, radius, d_radius=0, num_frames=30):
        ''' ele, azi (deg): target elevation and azimuth, 
        radius: target radius, used to determine the target position 
        d_radius: change of radius
        num_frames: new frames added (Note a start frame is already added !!)
        '''
        target = self.curcam.T - radius * self.curcam.R[:,2]
        # let azi here in [0, 360)
        ele, azi, _ = self.curcam.get_orbit(target)
        if azi < 0:
            azi += 360
        step_e = (tar_ele - ele) / num_frames
        step_a = (tar_azi - azi) / num_frames
        step_r = d_radius / num_frames
        for i in range(num_frames):
            ele += step_e
            azi += step_a
            radius += step_r
            self.curcam.set_orbit(ele, azi, radius, target)
            self.current_traj.append(self.curcam)
            self.curcam = self.curcam.copy()
        return self

    def move_orbit_to_tmp(self, tar_ele, tar_azi, radius, d_radius=0, targetdx=0.5, targetdz=0.9, num_frames=30):
        ''' ele, azi (deg): target elevation and azimuth, 
        radius: target radius, used to determine the target position 
        d_radius: change of radius
        num_frames: new frames added (Note a start frame is already added !!)
        '''
        target = self.curcam.T - radius * self.curcam.R[:,2]
        # let azi here in [0, 360)
        ele, azi, _ = self.curcam.get_orbit(target)
        if azi < 0:
            azi += 360
        amod = RotationModel(max_angle=tar_azi, t_total=num_frames, t_middle=0)
        step_e = (tar_ele - ele) / num_frames
        #step_a = (tar_azi - azi) / num_frames
        step_r = d_radius / num_frames
        for i in range(num_frames):
            ele += step_e
            azi = amod.degfromT(i)
            radius += step_r
            self.curcam.set_orbit(ele, azi, radius, target)
            self.current_traj.append(self.curcam)
            self.curcam = self.curcam.copy()
        return self

    def focal_change(self,tar_f,num_frames=30):
        ''' tar_f: target focal length
        '''
        step_f = (tar_f - self.curcam.f) / num_frames
        for i in range(num_frames):
            self.curcam.f += step_f
            self.current_traj.append(self.curcam)
            self.curcam = self.curcam.copy()
        return self

    def reinterpolate(self,num_frames=None):
        """
            Reinterpolate the current trajectory
        """
        startcam = self.startcam
        name=self.curname
        if num_frames is None:
            num_frames = len(self.current_traj)-1
        final_c2w=self.current_traj[-1].getC2W()
        self.finish()
        return self.add_traj(startcam=startcam,name=name+'-reinterpolate').move_pose_to(final_c2w, num_frames=num_frames)
    
    def finish(self):
        if getattr(self, "startcam", None) is None:
            raise ValueError("No trajectory to finish")
        traj = self.current_traj
        self.current_traj = None
        self.startcam = None
        self.curname = None
        return traj
    
    def drop(self):
        traj = self.current_traj
        self.current_traj = None
        self.startcam = None
        del self.camtrajs[self.curname]
        self.curname = None
        return traj


    def getTraj(self, name):
        if name not in self.camtrajs:
            raise ValueError(f"Trajectory {name} not found, available: {list(self.camtrajs.keys())}")
        return self.camtrajs[name]

    def render_video(self, name, renderer, output_path=None, fps=30):
        traj = self.getTraj(name)
        CamPlanner._render_video(renderer, traj, output_path, fps)
    
    def _render_video_batched(self, renderer, traj, output_path=None, fps=30, batchsize=100):
        '''return a list of frames [HW3] tensor (cpu)'''
        num_batch = len(traj) // batchsize
        frames = []
        for i in range(num_batch):
            batch = traj[i*batchsize:(i+1)*batchsize]
            new_frames = CamPlanner._render_video(renderer, batch, output_path=None, fps=fps)
            frames.extend([f.cpu() for f in new_frames])
        if output_path is not None:
            import imageio
            imageio.mimsave(output_path, frames, fps=fps)
        return frames

    def _render_video(renderer, traj, output_path=None, fps=30,mask=False):
        '''return a list of frames [HW3] tensor (cuda)'''
        frames = []
        for mcam in tqdm(traj, disable=CamPlanner.suppress_tqdm):
            if renderer.__class__.__name__ == 'PcdMgr':
                res = renderer.render(mcam, mask = mask)
                if mask:
                    res = res.squeeze()
                else:
                    res = res.permute((0,2,3,1)).squeeze(0)
            elif renderer.__class__.__name__ == 'GaussianMgr':
                # this is rendered with rgbd, gs result [H,W,3]
                res = renderer.render(mcam)
                if mask:
                    res = res[2].squeeze()
                else:
                    res = res[0]
            else:
                raise ValueError("Invalid renderer {}".format(renderer.__class__.__name__))
                
            frames.append(res)
        if output_path is not None:
            import imageio
            if mask:
                frames_uint8 = [(frame.unsqueeze(-1) * 255).cpu().to(dtype = torch.uint8) for frame in frames]
            else:
                frames_uint8 = [(frame * 255).cpu().to(dtype = torch.uint8) for frame in frames]
            imageio.mimsave(output_path, frames_uint8, fps=fps, macro_block_size = 4)
        return frames
    

class Mcam:
    default_f = 430.7958
    def __init__(self):
        self.H: int = 288
        self.W: int = 512
        self.f: float = Mcam.default_f
        self.c: tuple = (256, 144)
        self.R: np.ndarray = np.eye(3)
        self.T: np.ndarray = np.zeros(3)

    def __repr__(self):
        self.check_sanity()
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))
    
    def check_sanity(self):
        if not isinstance(self.H, int) or not isinstance(self.W, int):
            raise ValueError("H and W should be integers")
        if not isinstance(self.f, float):
            raise ValueError(f"f should be float, got {type(self.f)}")
        if not isinstance(self.c, tuple) or len(self.c) != 2:
            raise ValueError(f"c should be tuple of 2: {self.c}")
        if not isinstance(self.R, np.ndarray) or self.R.shape != (3, 3):
            raise ValueError(f"R should be [3,3] np.ndarray, got {self.R.shape}")
        if not isinstance(self.T, np.ndarray) or self.T.shape != (3,):
            raise ValueError(f"T should be [3] np.ndarray, got {self.T.shape}")
        
    def set_C2W(self, C2W):
        self.R = C2W[:3,:3]
        self.T = C2W[:3,3]
        return self

    def set_default_c(self):
        self.c = (self.W // 2, self.H // 2)

    @classmethod
    def set_default_f(cls, f):
        cls.default_f = f
        
    def set_size(self, H, W):
        fov = self.getfov('x')
        self.H, self.W = H, W
        self.set_default_c()
        self.setfov(fov,'x')
        return self

    def set_cam(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if not hasattr(self, key):
                raise ValueError(f"Invalid key {key}")
            setattr(self, key, value)
        return self

    def set_orbit(self, elevation, azimuth, radius, target=None):
        pose = orbit_camera(elevation, azimuth, radius, target=target)
        self.R = pose[:3,:3]
        self.T = pose[:3,3]
        return self
    
    def set_orbit_inplace(self, elevation, azimuth):
        ''' just rotate the camera, no change in position'''
        pose = orbit_camera(elevation, azimuth, radius=1, target=None)
        self.R = pose[:3,:3]
        return self

    def getK(self):
        return np.array([
            [self.f, 0, self.c[0]],
            [0, self.f, self.c[1]],
            [0, 0, 1]], dtype=np.float32).copy()
    
    def getC2W(self):
        RT = np.concatenate([self.R,self.T[:,None]],axis=1)
        RT = np.concatenate([RT,np.array([[0,0,0,1]])],axis=0)
        return RT.astype(np.float32).copy()
    
    def setC2W(self, RT):
        self.R = RT[:3,:3]
        self.T = RT[:3,3]
        return self
    
    def getC2W_RDF(self):
        R= np.stack([self.R[:, 0], -self.R[:, 1], -self.R[:, 2]], 1)
        RT = np.concatenate([R,self.T[:,None]],axis=1)
        RT = np.concatenate([RT,np.array([[0,0,0,1]])],axis=0)
        return RT.astype(np.float32)
    
    def getW2C(self):
        return np.linalg.inv(self.getC2W()).astype(np.float32)
    
    def copy(self):
        newcam = Mcam()
        newcam.H = self.H
        newcam.W = self.W
        newcam.f = self.f
        newcam.c = self.c
        newcam.R = self.R.copy()
        newcam.T = self.T.copy()
        return newcam
    
    def get_orbit(self, target=None):
        '''
        target=None: target is the camera looking at now
        azimuth: -180 - 180
        '''
        if target is None:
            target = self.T - np.array(self.R[:,2])
        radius = np.linalg.norm(self.T - target)
        if radius == 0:
            return 0, 0, 0
        elevation = np.arcsin((self.T[1] - target[1]) / radius)
        azimuth = np.arctan2((self.T[0] - target[0]), (self.T[2] - target[2]))
        elevation = np.rad2deg(elevation)
        azimuth = np.rad2deg(azimuth)
        return elevation, azimuth, radius
    
    def getfov(self, axis="x"):
        if axis == "x":
            return np.rad2deg(2 * np.arctan(self.W / (2 * self.f)))
        elif axis == "y":
            return np.rad2deg(2 * np.arctan(self.H / (2 * self.f)))
        else:
            raise ValueError("Invalid axis, only support x/y")
        
    def setfov(self, fov, axis):
        if axis == "x":
            self.f = self.W / (2 * np.tan(np.deg2rad(fov) / 2))
        elif axis == "y":
            self.f = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        else:
            raise ValueError("Invalid axis, only support x/y")
        return self


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
# [!] made changes here: positive y now
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    zero_radius = (radius == 0)
    if zero_radius:
        radius = 1
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    if zero_radius:
        campos = target
    T[:3, 3] = campos
    return T
