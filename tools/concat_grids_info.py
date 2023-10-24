from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import os
import sys
sys.path.append(os.getcwd())
import utils.utils_nyu as NyuUtils
import utils.utils_image as ImageUtils
colour_map_np = NyuUtils.nyu40_colour_code

from confs.path import lis_name_scenes
method_name = 'deeplab_ce_sv/uncertainty_1'
data_mode = 'train'

H , W = 240, 320

for scene_name in tqdm(lis_name_scenes, desc='processing scene...'):
    exps_dir = f'../exps/indoor/neus/{method_name}/{scene_name}'
    grids_info_dir = os.path.join(exps_dir, 'image_valiate_semantics')
    grids_info_list = sorted(glob(os.path.join(grids_info_dir, '*.png')))

    iter_list = np.unique([os.path.basename(grids_info)[0:8] for grids_info in grids_info_list])
    N_img = int(len(grids_info_list)/len(iter_list))
    path_dir = os.path.join(exps_dir, 'concat_validate')
    os.makedirs(path_dir, exist_ok=True)
    for idx in range(N_img):
        lis_img = []

        for iter_idx in range(len(iter_list)):
            lis = cv2.imread(grids_info_list[idx + iter_idx*N_img])
            lis_img.append(lis)
        
        lis_img = np.stack(lis_img, axis=0)
        ImageUtils.write_image_lis(os.path.join(path_dir, os.path.basename(grids_info_list[idx])[9:]), lis_img, cat_mode='vertical')

            

