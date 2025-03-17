import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/InstantMesh'
sys.path.insert(0,reference)

from instantmesh_command import InstantMesh
from torchvision.utils import save_image
import open3d as o3d
from PIL import Image
import numpy as np
import torch

class Image2Mesh_Tool():
    def __init__(self):
        self.instant_mesh = InstantMesh()
        self.image_cache_path = "./cache/1.png"
        self.obj_cache_dir = "./cache/"

    @staticmethod
    def mesh2pcd(mesh, num_points):
        point_cloud = mesh.sample_points_uniformly(num_points)
        pts = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors) 
        colors = colors / 255.0
        pts6d = np.concatenate((pts, colors), axis=1)
        # do this to align with instant-mesh file read result
        #pts6d[:,1] = -pts6d[:,1]

        return pts6d

    def __call__(self, image, numpoints=30000):
        if isinstance(image, Image.Image):
            image.save(self.image_cache_path)
        elif isinstance(image, torch.Tensor):
            save_image(image, self.image_cache_path)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image.save(self.image_cache_path)
        elif isinstance(image, str):
            self.image_cache_path = image
        mesh_out = self.instant_mesh(self.image_cache_path, mesh_path=self.obj_cache_dir)
        vertices, faces, vertex_colors = mesh_out

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        pcd6d = self.mesh2pcd(mesh, numpoints)

        return pcd6d
