import open3d as o3d
import numpy as np
import cv2, os


def visualize_3d_points(points, colors):
    # 创建一个Open3D点云对象
    pcd = o3d.geometry.PointCloud()

    # 从Nx3 Numpy数组中设置点
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 颜色必须在0到1之间

    # 创建一个视窗并添加点云
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud('521030910030.ply', pcd)
    
    
if __name__ == '__main__':
    
    dir = 'full_res'
    
    # points = np.load(os.path.join(dir, 'pts.npy'))
    # colors = np.load(os.path.join(dir, 'crs.npy'))
    
    # visualize_3d_points(points, colors)
    
    Rs = np.load(os.path.join(dir, 'Rs.npy'))
    ts = np.load(os.path.join(dir, 'ts.npy'))
    
    with open('521030910030.txt', 'w') as f:
        for i in range(11):
            now = [Rs[i][0, 0], Rs[i][0, 1], Rs[i][0, 2], ts[i][0], 
                   Rs[i][1, 0], Rs[i][1, 1], Rs[i][1, 2], ts[i][1], 
                   Rs[i][2, 0], Rs[i][2, 1], Rs[i][2, 2], ts[i][2], 
                   0, 0, 0, 1]
            res = [str(_) for _ in now]
            f.write(' '.join(res)+'\n')