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
method_name_lis = ['deeplab_ce_sv/num_1', 'deeplab_ce_sv/prob_1']
data_mode = 'train'
iter = '00040000'
H , W = 240, 320

contrast_exps_dir = f'../exps/indoor/neus/contrast/contrast_deeplab_ce_sv'
os.makedirs(contrast_exps_dir, exist_ok=True)
with open(os.path.join(contrast_exps_dir, f'method_name_lis.txt'), 'w') as f:
    f.writelines(method_name_lis)

for scene_name in tqdm(lis_name_scenes, desc='processing scene...'):

    exps_dir = f'../exps/indoor/neus/{method_name_lis[0]}/{scene_name}'
    lis_exps_dir = os.path.join(contrast_exps_dir, scene_name)
    os.makedirs(lis_exps_dir, exist_ok=True)

    grids_info_dir = os.path.join(exps_dir, 'image_valiate_semantics')
    grids_info_list = sorted(glob(os.path.join(grids_info_dir, f'{iter}_*.png')))
    N_img = len(grids_info_list)

    for idx in tqdm(range(N_img), desc = 'concatting'):
        lis_img = []
        for method_name in method_name_lis:
            exps_dir = f'../exps/indoor/neus/{method_name}/{scene_name}'
            grids_info_dir = os.path.join(exps_dir, 'image_valiate_semantics')
            grids_info_list = sorted(glob(os.path.join(grids_info_dir, f'{iter}_*.png')))

            img = cv2.imread(grids_info_list[idx])
            lis_img.append(img)
        
        lis_img = np.stack(lis_img, axis=0)
        ImageUtils.write_image_lis(os.path.join(lis_exps_dir, os.path.basename(grids_info_list[idx])[9:]), lis_img, cat_mode='vertical')




            

