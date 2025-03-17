import numpy as np
import open3d as o3d
import trimesh
from PIL import Image  # 导入 Pillow 库

def glb_mesh_to_point_cloud(glb_input_path, output_path, num_points=100000):
    """
    将 Blender 导出的 GLB 文件转换为点云并保存到 output_path。
    
    参数:
      glb_input_path -- GLB 文件路径
      output_path    -- 点云输出文件路径，例如 "output.ply"
      num_points     -- 采样点数，默认 100000
    """
    # 以场景方式加载文件，保留场景信息
    scene = trimesh.load(glb_input_path, force='scene')
    
    # 遍历场景中所有几何体，正确应用全局变换
    meshes = []
    for name, mesh in scene.geometry.items():
        try:
            # 尝试获取从 world 到该 mesh 的全局变换
            T = scene.graph.get(name)[0]
        except Exception:
            # print(f"警告: 无法找到从 world 到 {name} 的转换，使用单位矩阵。")
            T = np.eye(4)
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(T)
        meshes.append(mesh_copy)
    
    if len(meshes) == 0:
        raise ValueError("未在 GLB 文件中找到 mesh 几何体")
    
    # 合并所有网格
    combined = trimesh.util.concatenate(meshes)
    
    # 根据 combined.visual.kind 提取颜色信息
    if combined.visual.kind == 'vertex' and \
       hasattr(combined.visual, 'vertex_colors') and \
       combined.visual.vertex_colors is not None:
        # 采样时返回 face 索引，有助于定位颜色
        points, face_indices = combined.sample(num_points, return_index=True)
        # 获取采样点所在面的顶点颜色数组（形状: [n, 3, 3]）
        face_vcolors = combined.visual.vertex_colors[combined.faces[face_indices], :3].astype(np.float64)
        # 简单起见，这里取每个面第一个顶点的颜色作为该采样点的颜色，归一化到 [0, 1]
        colors = face_vcolors[:, 0, :] / 255.0
    elif combined.visual.kind == 'texture':
        try:
            # 尝试将纹理映射转换为顶点颜色
            vcolors = combined.visual.to_color().vertex_colors[:, :3].astype(np.float64) / 255.0
            points, face_indices = combined.sample(num_points, return_index=True)
            # 同样取第一个顶点的颜色
            colors = vcolors[combined.faces[face_indices][:,0]]
        except Exception as e:
            print("纹理颜色提取失败，使用默认颜色：", e)
            points = combined.sample(num_points)
            colors = np.tile(np.array([[1, 1, 1]]), (points.shape[0], 1))
    else:
        # 若无颜色信息，统一使用默认白色
        points = combined.sample(num_points)
        colors = np.tile(np.array([[1, 1, 1]]), (points.shape[0], 1))
    
    #### 注意!!!临时修改!!!
    points[:, [1,2]] = points[:, [2,1]]
    points[:, [1]] = -points[:, [1]]
    colors = colors[points[:,2]<5]
    points = points[points[:,2]<5]

    points[:,:3]*=0.05
    points[:,0]+=0.1
    points[:,2]-=0.3
    
    # 创建 Open3D 点云对象，并赋予采样点和颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    success = o3d.io.write_point_cloud(output_path, pcd)
    if success:
        print("点云已成功保存至：", output_path)
    else:
        print("点云保存失败，请检查输出路径或文件权限。")
    
    return np.hstack((points, colors))



def downsample_point_cloud(points, voxel_size=0.01):
    """
    对点云进行下采样

    参数:
    points -- 点云数据 (N, 6) 包含位置和颜色
    voxel_size -- 体素网格的大小

    返回:
    下采样后的点云数据
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])

    # 进行体素网格下采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 提取下采样后的点和颜色
    downsampled_points = np.hstack((np.asarray(downsampled_pcd.points), np.asarray(downsampled_pcd.colors)))

    return downsampled_points