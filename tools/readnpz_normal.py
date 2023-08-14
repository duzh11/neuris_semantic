import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing 
from utils.utils_geometry import get_world_normal
import torch

def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose

scene_num='0616_00'
img_num='0870'
normal_pred_dir='/home/du/Proj/NeuRIS/exps/indoor/neus/scene'+scene_num+'/neuris/normal_render/'
normal_gt_dir='/home/du/Proj/NeuRIS/Data/dataset/indoor/scene'+scene_num+'/pred_normal/'
pose_dir='/home/du/Proj/NeuRIS/Data/dataset/indoor/scene'+scene_num

pred_name='00160000_'+img_num+'_reso1.npz'
gt_name=img_num+'.npz'
pred_f=os.path.join(normal_pred_dir,pred_name)
gt_f=os.path.join(normal_gt_dir,gt_name)
path_cam=os.path.join(pose_dir,'cameras_sphere.npz')
camera_dict = np.load(path_cam)

world_mat = camera_dict['world_mat_%d' % int(int(img_num)/10)].astype(np.float32)
scale_mat = camera_dict['scale_mat_%d'% int(int(img_num)/10)].astype(np.float32)
P = world_mat @ scale_mat
P = P[:3, :4]
_, pose = load_K_Rt_from_P(P)
ex_i= np.linalg.inv(pose)

pred_npz = np.load(pred_f)['arr_0']
gt_npz = np.load(gt_f)['arr_0']

# 因为neuris在导入normal时加上了负号
gt_new = -get_world_normal(gt_npz.reshape(-1, 3), ex_i).reshape(gt_npz.shape[0],gt_npz.shape[1],3)

norm_pred=np.linalg.norm(pred_npz, axis=-1, ord=2,keepdims=True)
norm_gt=np.linalg.norm(gt_new, axis=-1, ord=2,keepdims=True)
img_pred=(pred_npz/norm_pred)
img_gt=(gt_new/norm_gt)

img_pred=(((img_pred + 1) * 0.5).clip(0,1) * 255)
img_gt=(((img_gt + 1) * 0.5).clip(0,1) * 255)

cv2.imwrite(normal_pred_dir+f'00160000_{img_num}_reso1.png',(img_pred.astype(np.uint8))[...,::-1])
cv2.imwrite(normal_gt_dir+f'{img_num}_render.png',(img_gt.astype(np.uint8))[...,::-1])

print('complete')

