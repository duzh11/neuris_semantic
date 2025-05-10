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
method_name_lis = ['deeplab_ce_sv/num_1', 'deeplab_ce_sv/prob_1', \
                   'deeplab_ce_sv/uncertainty_exp_validate', 'deeplab_ce_sv/uncertainty_exp_prob_validate']
data_mode = 'train'
iter = '00040000'
H , W = 240, 320

contrast_exps_dir = './exps/indoor/neus/contrast/contrast_deeplab_ce_sv'
os.makedirs(contrast_exps_dir, exist_ok=True)

for scene_name in tqdm(lis_name_scenes, desc='processing scene...'):
    lis_exps_dir = os.path.join(contrast_exps_dir, scene_name)
    os.makedirs(lis_exps_dir, exist_ok=True)

    grids_dir = f'./Data/dataset/indoor/{scene_name}/grids/{data_mode}'
    grids_list = sorted(glob(os.path.join(grids_dir, 'felzenszwalb_100_1_50_a_vis', '*.png')))

    exps_dir = f'./exps/indoor/neus/{method_name_lis[0]}/{scene_name}'
    uncertrainty_dir = os.path.join(exps_dir, 'sem_uncertainty_vis', data_mode, 'fine')
    uncertrainty_list = sorted(glob(os.path.join(uncertrainty_dir, f'{iter}_*.png')))
    N_img = len(uncertrainty_list)

    lis_img_all=[]
    for idx in tqdm(range(N_img), desc = 'concatting'):
        lis_img = []
        lis_img.append(cv2.resize( cv2.imread(grids_list[idx]), (W, H)))
        lis_img.append(cv2.resize( cv2.imread(uncertrainty_list[idx]), (W, H)))

        for method_name in method_name_lis:
            exps_dir = f'./exps/indoor/neus/{method_name}/{scene_name}'
            
            grids_info_dir = os.path.join(exps_dir, 'girds_info', data_mode, 'fine')
            grids_info_list = sorted(glob(os.path.join(grids_info_dir, f'{iter}_*.png')))

            img = cv2.imread(grids_info_list[idx])
            lis_img.append(img)
            lis_img.append(255 * np.ones((H, 10, 3)).astype('uint8'))
        
        lis_img = np.concatenate(lis_img, axis=1)
        lis_img_all.append(lis_img)
    
    (height, width, _) = lis_img_all[0].shape
    video_name = f'{lis_exps_dir}/{scene_name}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videowriter = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for image in lis_img_all:
        videowriter.write(image.astype(np.uint8))

    videowriter.release()


    

