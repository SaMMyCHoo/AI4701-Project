# perform 3D reconstruction using PnP
import cv2
import numpy as np

def pnp_recon(object_points, image_points, camera_matrix):

    # 畸变系数，这里假设没有畸变
    dist_coeffs = np.zeros(5)

    # 调用solvePnP
    ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    t = tvec

    return R, t
