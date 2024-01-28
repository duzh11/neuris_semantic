import sys, os
sys.path.append(os.getcwd())
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

from utils.utils_image import read_images, write_image, write_images
from utils.utils_geometry import get_world_normal
import models.dataset as dataset
import utils.utils_image as ImageUtils

data_root = '../Data/dataset/indoor'
from confs.path import lis_name_scenes

for scene in lis_name_scenes:
    for data_mode in ['train', 'test']:
        data_dir = os.path.join(data_root, scene)
        normal_dir = os.path.join(data_dir, f'normal/{data_mode}/pred_normal/*.png') 
        normal_list = sorted(glob(normal_dir))
        h_img, w_img, _ = cv2.imread(normal_list[0]).shape
        n_images = len(normal_list)

        path_cam = os.path.join(data_dir, f'./cameras_sphere_{data_mode}.npz')  # cameras_sphere, cameras_linear_init
        camera_dict = np.load(path_cam)
        print(f'Load camera dict: {path_cam.split("/")[-1]}')

        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

        # loading camera para
        intrinsics_all = []
        pose_all = []
        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat #w2i
            P = P[:3, :4]
            intrinsics, pose = dataset.load_K_Rt_from_P(None, P) # n2w
            intrinsics_all.append(intrinsics)
            pose_all.append(pose)
        
        # convert normal c2w
        normal_w_dir = os.path.join(data_dir, f'normal/{data_mode}/pred_normal_world') 
        os.makedirs(normal_w_dir, exist_ok=True)
        normals_npz, stems_normal = read_images(f'{data_dir}/normal/{data_mode}/pred_normal', target_img_size=(w_img, h_img), img_ext='.npz')
        assert len(normals_npz) == n_images
        for i in tqdm(range(n_images)):
            normal_img_curr = normals_npz[i]
    
            # transform to world coordinates
            ex_i = np.linalg.inv(pose_all[i]) #w2c
            normal_w = -get_world_normal(normal_img_curr.reshape(-1, 3), ex_i).reshape(h_img, w_img,3)
            normal_w_vis=(((normal_w + 1) * 0.5).clip(0,1) * 255).astype(np.uint8)

            np.savez(os.path.join(normal_w_dir, os.path.basename(normal_list[i]).split('.')[0]+'.npz'), normal_w)
            ImageUtils.write_image(os.path.join(normal_w_dir, os.path.basename(normal_list[i])), (normal_w_vis.astype(np.uint8))[...,::-1])

