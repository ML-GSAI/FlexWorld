import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/TRELLIS'
sys.path.insert(0,reference)

from trellis_command import TRELLIS
from torchvision.utils import save_image
import open3d as o3d
from PIL import Image
import numpy as np
import torch


class Image2Mesh_Tool():
    def __init__(self):
        self.trellis = TRELLIS()
        self.image_cache_path = "./cache/1.png"
        self.obj_cache_dir = "./cache/"
    
    
    @staticmethod
    def gs2pcd(gs):
        pts = gs.get_xyz.detach().cpu().numpy()
        sh = gs.get_features.transpose(1, 2).view(-1, 3, (gs.sh_degree+1)**2).cpu().numpy()
        C0 = 0.28209479177387814
        colors = C0 * sh[..., 0] + 0.5
        pts6d = np.concatenate((pts, colors), axis=1)
        return pts6d

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image.save(self.image_cache_path)
        elif isinstance(image, torch.Tensor):
            save_image(image, self.image_cache_path)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image.save(self.image_cache_path)
        elif isinstance(image, str):
            self.image_cache_path = image
        output = self.trellis(self.image_cache_path, mesh_path=self.obj_cache_dir)
        pcd6d = self.gs2pcd(output)

        return pcd6d
