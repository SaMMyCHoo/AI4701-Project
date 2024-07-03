# perform bundle adjustment here

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from open3d import *
import copy
from collections import Counter



# from bundle_adjustment.py
def rotate(points, rot_vecs):
    '''
    Rotate points by given rotation vectors, Rodrigues' rotation formula is used.
    Input: 
        points:     
        rot_vecs:
    Output:
        rotated points
    '''
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    '''
    Convert 3-D points to 2-D by projecting onto images.
    '''
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj=points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    '''
    get residuals
    '''
    # recover cameras params and points 3d
    camera_params=params[:n_cameras*6].reshape((n_cameras,6))
    points_3d=params[n_cameras*6:].reshape((n_points,3))
    
    # reproject 3d points
    points_proj = project(points_3d[point_indices], camera_params[camera_indices]) 
    residual=points_proj-points_2d 
    residual=np.dot(K[:2,:2],residual.T).T 
    return residual.ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    '''
    build sparse jacobian
    '''
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3 
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6): 
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1 

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1 
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    return A



def bundle_adjustment(camera_params, points_2d, points_3d, camera_indices, point_indices, intrinsic):
    '''
    做BA
    获得BA的起点x0, 矩阵A, 解一个最小二乘优化问题
    # 详见 https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    '''
    n_cameras = len(camera_params) 
    n_points = len(points_3d)
    
    if(points_2d.shape[1] == 2):
        points_2d=np.hstack((points_2d, np.ones((points_2d.shape[0],1)))) # 转齐次坐标
    points_2d = np.dot(np.linalg.inv(intrinsic), points_2d.T).T # 转归一化相机平面坐标
    points_2d = points_2d[:,:2] # 转非齐次坐标
    
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(fun, x0,jac='3-point',jac_sparsity=A, verbose=1, x_scale='jac', ftol=1e-3, method='trf',loss='soft_l1',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, intrinsic))
    return res