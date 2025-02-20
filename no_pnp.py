import open3d as o3d
import numpy as np
import cv2

from feature_matching import match_features
from initial_recon import compute_matrices_and_pose, triangulate_matches
from pnp_recon import pnp_recon
from bundle_adjustment import bundle_adjustment


def calculate_reprojection_error(X, x_obs, K, R, t):
    # Ensure X is in the correct shape for OpenCV
    X = np.array([X], dtype=np.float32)

    # Convert rotation matrix to a rotation vector
    rvec, _ = cv2.Rodrigues(R)

    # Ensure translation vector is the correct shape
    tvec = np.array(t, dtype=np.float32).reshape(3, 1)

    # Project the 3D points to 2D
    x_proj, _ = cv2.projectPoints(X, rvec, tvec, K, distCoeffs=None)

    # Calculate the Euclidean distance between the projected and observed points
    error = np.linalg.norm(x_obs - x_proj.squeeze())
    return error

def visualize_3d_points(points, colors):
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)

    # 过滤掉那些在任一维度上偏差过大的点
    valid_points = np.all(np.abs(points - mean) < 2 * std, axis=1)
    filtered_points = points[valid_points]
    filtered_colors = colors[valid_points]

    print(len(valid_points))
    
    np.save('pts.npy', filtered_points)
    np.save('crs.npy', filtered_colors)
    # 创建一个Open3D点云对象
    pcd = o3d.geometry.PointCloud()

    # 从Nx3 Numpy数组中设置点
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)  # 颜色必须在0到1之间

    # 创建一个视窗并添加点云
    o3d.visualization.draw_geometries([pcd])

def create_transformation_matrix(R, t):
    """根据旋转矩阵R和平移向量t创建变换矩阵T"""
    T = np.eye(4)  # 创建一个4x4的单位矩阵
    T[:3, :3] = R  # 填充旋转矩阵
    T[:3, 3] = t   # 填充平移向量
    return T

def concatenate_transformations(R1, t1, R2, t2):
    """
    给定两个变换（R1, t1）和（R2, t2），计算并返回合成后的变换（R, t）
    其中 (R1, t1) 是第一个变换矩阵（例如从坐标系1到坐标系2）
         (R2, t2) 是第二个变换矩阵（例如从坐标系2到坐标系3）
    返回的 (R, t) 是从第一个坐标系直接到第三个坐标系的变换
    """
    # 创建两个变换矩阵
    T1 = create_transformation_matrix(R1, t1)
    T2 = create_transformation_matrix(R2, t2)
    
    # 矩阵乘法得到合成变换
    T = np.dot(T1, T2)
    
    # 提取合成变换的旋转矩阵和平移向量
    R = T[:3, :3]
    t = T[:3, 3]
    
    return R, t

def main():
    
    poses = []
    
    colors3D = []
    points3D = []
    points2D = []
    cameraIndices = []
    point2DIndices = []
    
    images = ['images/0000.png', 
              'images/0001.png', 
              'images/0002.png',
              'images/0003.png', 
              'images/0004.png',
              'images/0005.png', 
              'images/0006.png',
              'images/0007.png',
              'images/0008.png', 
              'images/0009.png',   
              'images/0010.png']
    
    image_data = [cv2.imread(img) for img in images]
    image_data = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_data]  # OpenCV uses BGR, convert to RGB
    
    K = np.array([[2759.48, 0, 1520.69],
                  [0, 2764.16, 1006.81],
                  [0, 0, 1]])

    # 首先用前两张图计算出3d坐标
    res = match_features(images[0], images[1])
    pts1 = res[1]
    pts2 = res[2]
    
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    poses.append((R0, t0))
    R, t = compute_matrices_and_pose(pts1, pts2, K)
    poses.append((R, t))
    p3d = triangulate_matches(R0, t0, R, t, pts1, pts2, K)
    
    # print(R)
    # print()
    
    # print(p3d.shape)
    for i in range(len(p3d)):
        tmp = p3d[i]
        color = image_data[0][int(pts1[i][1]), int(pts1[i][0])]
        
        colors3D.append(color)        
        points3D.append(tmp)
        
        points2D.append(pts1[i])
        cameraIndices.append(0)
        point2DIndices.append(len(points3D) - 1)

        points2D.append(pts2[i])
        cameraIndices.append(1)
        point2DIndices.append(len(points3D) - 1)
    
    #清理门户  
    points = np.array(points3D)
    colors = np.array(colors3D)

    # 过滤掉那些在任一维度上偏差过大的点
    valid_points = np.ones(len(points3D), dtype=bool)

    for j in range(len(points2D)):
        x = points3D[point2DIndices[j]]
        x_ob = points2D[j]
        R, t = poses[cameraIndices[j]]
        err = calculate_reprojection_error(x, x_ob, K, R, t)
        if err > 20:
            valid_points[point2DIndices[j]] = False
    
    # 创建一个从旧索引到新索引的字典映射，不存在的索引置为 -1
    index_mapping_dict = {}

    new_index = 0
    for old_index, is_valid in enumerate(valid_points):
        if is_valid:
            index_mapping_dict[old_index] = new_index
            new_index += 1
        else:
            index_mapping_dict[old_index] = -1
    
    #同步删除剩下的值
    for j in range(len(points2D)):
            point2DIndices[j] = index_mapping_dict[point2DIndices[j]]
    
    # 创建一个布尔数组，初值为 False，长度与 point2DIndices 相同
    valid = np.zeros(len(point2DIndices), dtype=bool)

    # 更新 valid 数组，检查每个旧索引是否映射到有效的新索引
    for j in range(len(point2DIndices)):
        if point2DIndices[j] != -1:
            valid[j] = True    

    # print(len(valid))
    
    colors3D = np.array(colors3D)[valid_points]
    points3D = points[valid_points]
    cameraIndices = np.array(cameraIndices)[valid]
    point2DIndices = np.array(point2DIndices)[valid]
    points2D = np.array(points2D)[valid]    

    colors3D = list(colors3D)
    points3D = list(points3D)
    cameraIndices = list(cameraIndices)
    point2DIndices = list(point2DIndices)
    points2D = list(points2D)
    
    
    
    # 接下来，依次加入每一个相机，完成匹配，pnp，ba
    for i in range(2, 11):
        
        res = match_features(images[i - 1], images[i])
        pts1 = res[1]
        pts2 = res[2]
        
        R, t = compute_matrices_and_pose(pts1, pts2, K)
        R0, t0 = poses[i - 1]
        R, t = concatenate_transformations(R0, t0, R, t)
        poses.append((R, t))
        
        # print(R)
        # print()
        
        #拿到位姿后首先更新可用的点
        for j in range(i):
            R0, t0 = poses[j]
            R, t = poses[i]
            res = match_features(images[j], images[i])
            pts1 = res[1]
            pts2 = res[2]
            
            if(len(pts1) == 0):
                continue
            
            #三角定位
            p3d = triangulate_matches(R0, t0, R, t, pts1, pts2, K)
            for k in range(len(p3d)):
                color = image_data[j][int(pts1[k][1]), int(pts1[k][0])]
                tar = p3d[k]
                #首先判断这个点是不是已经在当前点云里面了，这个用2d的去找，不然误差太大
                tmp = pts1[k]
                done = False
                for l in range(len(points2D)):
                    if cameraIndices[l] != j:
                        continue
                    now = points2D[l]
                    dis = np.linalg.norm(tmp - now)
                    if dis < 0.00001:
                        #找到了，那么直接把这个2d点指到这个点就可以
                        done = True
                        points2D.append(pts2[k])
                        cameraIndices.append(i)
                        point2DIndices.append(point2DIndices[l])
                        break
                if done:
                    continue
                else:
                    #说明是新的坐标
                    points3D.append(tar)
                    colors3D.append(color)
                    points2D.append(tmp)
                    cameraIndices.append(j)
                    point2DIndices.append(len(points3D) - 1)

                    points2D.append(pts2[k])
                    cameraIndices.append(i)
                    point2DIndices.append(len(points3D) - 1)
        
        #ba优化前我们首先删除outlier，首先通过3sigma原则筛选
        
        points = np.array(points3D)
        # colors = np.array(colors_3d)

        # 过滤掉那些在任一维度上偏差过大的点
        valid_points = np.ones(len(points3D), dtype=bool)

        for j in range(len(points2D)):
            x = points3D[point2DIndices[j]]
            x_ob = points2D[j]
            R, t = poses[cameraIndices[j]]
            err = calculate_reprojection_error(x, x_ob, K, R, t)
            if err > 20:
                valid_points[point2DIndices[j]] = False
        
        # 创建一个从旧索引到新索引的字典映射，不存在的索引置为 -1
        index_mapping_dict = {}

        new_index = 0
        for old_index, is_valid in enumerate(valid_points):
            if is_valid:
                index_mapping_dict[old_index] = new_index
                new_index += 1
            else:
                index_mapping_dict[old_index] = -1
        
        #同步删除剩下的值
        for j in range(len(points2D)):
                point2DIndices[j] = index_mapping_dict[point2DIndices[j]]
        
        # 创建一个布尔数组，初值为 False，长度与 point2DIndices 相同
        valid = np.zeros(len(point2DIndices), dtype=bool)

        # 更新 valid 数组，检查每个旧索引是否映射到有效的新索引
        for j in range(len(point2DIndices)):
            if point2DIndices[j] != -1:
                valid[j] = True    
    
        # print(len(valid))
        colors3D = np.array(colors3D)[valid_points]
        points3D = points[valid_points]
        cameraIndices = np.array(cameraIndices)[valid]
        point2DIndices = np.array(point2DIndices)[valid]
        points2D = np.array(points2D)[valid]
        temp = points2D
        
        # if(points2D.shape[1] == 2):
        #     points2D=np.hstack((points2D, np.ones((points2D.shape[0],1)))) # 转齐次坐标
        # points2D = np.dot(np.linalg.inv(K), points2D.T).T # 转归一化相机平面坐标
        # points2D = points2D[:,:2] # 转非齐次坐标
        
        # cameraArray = []
        # for j in range(i + 1):
        #     R, t = poses[j]
        #     R, _ = cv2.Rodrigues(R)
        #     camera = np.hstack((R.ravel(), t.ravel()))
        #     cameraArray.append(camera)
        # cameraArray = np.array(cameraArray)
        
        # final = bundle_adjustment(cameraArray, points2D, points3D, cameraIndices, point2DIndices, K).x
        
        # points3D = final[(i + 1) * 6 :].reshape(-1, 3)
        
        # final = final[: (i + 1) * 6].reshape(i + 1, 6)
        
        # poses = []
        
        # for j in range(i + 1):
        #     pose = final[j]
        #     R = cv2.Rodrigues(pose[:3])[0]
        #     t = pose[3:]
        #     poses.append((R, t))

        # 复原坐标
        points3D = list(points3D)
        cameraIndices = list(cameraIndices)
        point2DIndices = list(point2DIndices)
        points2D = list(temp)
        colors3D = list(colors3D)

    # print(len(points3D))
    visualize_3d_points(np.array(points3D), np.array(colors3D))
    
main()