import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def add_axes(ax, points):
    # 提取 x, y, z 坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    #
    #points = points * 1.05
    # points[:, 2] -= 0.5 f74
    points[:, 0] = points[:, 0] + 0.2
    #points[:, 2] = points[:, 2] + 0.3


    # 插值生成平滑曲线
    t = np.arange(len(points))  # 参数化曲线
    t_new = np.linspace(t.min(), t.max(), 300)  # 更密集的参数点
    spline = make_interp_spline(t, points, k=3)  # 三次样条插值
    smoothed_points = spline(t_new)

    # 提取平滑曲线的 x, y, z 坐标
    x_smooth = smoothed_points[:, 0]
    y_smooth = smoothed_points[:, 1]
    z_smooth = smoothed_points[:, 2]

    # 绘制平滑曲线
    color = np.array([200, 0, 0])/255.0
    ax.plot(x_smooth[:-2], y_smooth[:-2], z_smooth[:-2], '-', label="平滑曲线", lw=2.5, c=color)
    


    # 在曲线末端添加箭头
    arrow_start = smoothed_points[-2]  # 箭头起点
    arrow_end = smoothed_points[-1]    # 箭头终点
    arrow = Arrow3D(
        [arrow_start[0], arrow_end[0]],  # X 坐标
        [arrow_start[1], arrow_end[1]],  # Y 坐标
        [arrow_start[2], arrow_end[2]],  # Z 坐标
        mutation_scale=20,               # 箭头大小
        arrowstyle="->",                # 箭头样式
        color=color,                       # 箭头颜色
        lw=2,                            # 线宽
    )
    ax.add_artist(arrow)


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 18))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None  # plotly data traces
        # self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        # self.ax.set_xlabel('x')
        # self.ax.set_ylabel('y')
        # self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))
        

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical', label='Frame Indexes')

    def show(self):
        #plt.title('Camera Trajectory')
        plt.savefig("./cache/1.png",dpi=300)



def draw_camera_trajectory(visualizer, c2ws, outpath="./cache/1.png", hw_ratio=9/16, base_xval=0.05, zval=0.075, xyz_scale=2):
    base_xval = base_xval * xyz_scale
    zval = zval * xyz_scale

    for c2w in c2ws:
        c2w[[2,1],:] = c2w[[1,2],:] 
    
    for frame_idx, c2w in enumerate(c2ws):
        visualizer.extrinsic2pyramid(c2w, frame_idx / len(c2ws), hw_ratio=hw_ratio, base_xval=base_xval, zval=zval)

    pts3d = []
    for i in range(len(c2ws)):
        pts3d.append(c2ws[i][:3, 3])
    pts3d = np.stack(pts3d)
    print(pts3d)
    add_axes(visualizer.ax, pts3d)

    visualizer.colorbar(len(c2ws))
    #plt.savefig(outpath, dpi=300)
