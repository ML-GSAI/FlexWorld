from ops.utils.utils_3d import downsample_point_cloud

class Image2Pcd_Tool():
    def __init__(self, opt, device='cuda'):
        if opt.type_3dgen=='trellis':
            from ops.trellis import Image2Mesh_Tool
            self.model = Image2Mesh_Tool()
        else:
            from ops.instant import Image2Mesh_Tool
            self.model = Image2Mesh_Tool()
            
    def __call__(self, image, voxel_size=0.005):
        pcd6d = self.model(image)
        if voxel_size is not None:
            pcd6d = downsample_point_cloud(pcd6d, voxel_size=voxel_size)
        print(pcd6d.shape)
        return pcd6d