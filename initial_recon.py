# use epipolar geometry to find the fundamental matrix and the essential matrix
# use the essential matrix to find the relative pose between two cameras
# use the relative pose to triangulate 3D points
import cv2
import numpy as np
from feature_matching import match_features

def decompose_essential_matrix(E):
    """ 对本质矩阵进行SVD分解，并返回所有可能的R和t """
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, _, Vt = np.linalg.svd(E)

    # 保证特解是反射而非旋转
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    return [R1, R2], [t, -t]

def check_depth(R, t, pts1, pts2, K):
    num_positive_depth = 0
    # 创建相机矩阵P1和P2
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(-1, 1)))

    # 确保点坐标是(2, N)格式
    pts1 = np.asarray(pts1).T  # 假设 pts1 是一个列表，包含(N, 2)格式的点
    pts2 = np.asarray(pts2).T  # 假设 pts2 是一个列表，包含(N, 2)格式的点

    # 进行三角测量
    X = cv2.triangulatePoints(P1, P2, pts1, pts2)
    X = X / X[3]  # 归一化到齐次坐标

    # 检查每个点的深度
    for i in range(X.shape[1]):
        if X[2, i] > 0 and (R @ X[:3, i] + t)[2] > 0:
            num_positive_depth += 1

    return num_positive_depth


def compute_matrices_and_pose(pts1, pts2, K):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    E = np.transpose(K) @ F @ K

    # 获取R和t的所有可能组合
    Rs, ts = decompose_essential_matrix(E)

    max_depth = 0
    best_R = None
    best_t = None

    # 检查每一组解
    for R in Rs:
        for t in ts:
            depth = check_depth(R, t, pts1, pts2, K)
            if depth > max_depth:
                max_depth = depth
                best_R = R
                best_t = t
    
    return best_R, best_t

def triangulate_matches(R1, t1, R2, t2, pts1, pts2, K):
    
    # 创建相机的投影矩阵
    P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, t2.reshape(3, 1)))

    # 将点列表转换为合适的Numpy数组格式
    pts1_np = np.array(pts1).T  # 转换为2xN的数组
    pts2_np = np.array(pts2).T  # 转换为2xN的数组

    # 三角测量，计算3D点
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_np, pts2_np)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  # 将齐次坐标转换为3D坐标
    return points_3d.T  # 返回3xN的数组
