import os
import numpy as np
import cv2

from lu_vp_detect import VPDetection
from glob import glob

data_root = './Data/dataset/indoor'
scene_list = ['scene0435_02', 'scene0616_00']

CROP=16
def extract_vps(filename, index, intrinsics):
    img = cv2.imread(filename)

    fx = intrinsics[0,0]
    fy = intrinsics[1,1]
    cx = intrinsics[0,2]
    cy = intrinsics[1,2]
    # undistort
    kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([[0, 0, 0, 0, 0]])
    img = cv2.undistort(img, kmat, dist)
    # flip
    principal_point = cx, cy
    # about how to choose fx or fy, the author's answer is https://github.com/rayryeng/XiaohuLuVPDetection/issues/4
    focal_length = fx
    seed = 2020
    vpd = VPDetection(60, principal_point, focal_length, seed)
    vps = vpd.find_vps(img) 
    #assert np.isnan(vps).all() == False, print(vps)
    vpd.create_debug_VP_image(show_image=True, save_image='vps_vis_25/{}.jpg'.format(index)) 
    vps = np.vstack([vps, -vps]).astype(np.float32)

    return vps

for scene in scene_list:
    data_dir = os.path.join(data_root, scene)
    intrinsics = np.loadtxt(os.path.join(data_dir, 'intrinsic_color_crop1248_resize640.txt'))
    
    for data_mode in ['train', 'test']:
        img_dir = sorted(glob(os.path.join(data_dir, f'image/{data_mode}/*.png')))
        for index, files in enumerate(img_dir):
            vps = extract_vps(files, index, intrinsics)