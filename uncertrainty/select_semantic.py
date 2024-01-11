import os, sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
from glob import glob
import utils.utils_nyu as NyuUtils
colour_map_np = NyuUtils.nyu40_colour_code

# from confs.path import lis_name_scenes
lis_name_scenes=['scene0378_00','scene0435_02'] 

Data_dir = '../Data/dataset/indoor'
for scene_name in lis_name_scenes:
    for data_mode in ['train', 'test']:
        img_dir = os.path.join(Data_dir, scene_name, f'image/{data_mode}')
        semantic_dir = os.path.join(Data_dir, scene_name, f'semantic/{data_mode}', 'deeplab')
        grids_dir = os.path.join(Data_dir, scene_name, f'grids/{data_mode}', 'SAM')
        semantic_score_dir = os.path.join(Data_dir, scene_name, f'semantic_uncertainty/{data_mode}/deeplab', 'viewALNeighbour')
        
        img_lis = sorted(glob(f'{img_dir}/*.png'))
        semantic_lis = sorted(glob(f'{semantic_dir}/*.png'))
        grids_lis = sorted(glob(f'{grids_dir}/*.png')) 
        valid_score_lis = sorted(glob(f'{semantic_score_dir}/valid_*.npz'))
        semantic_score_lis = sorted(glob(f'{semantic_score_dir}/viewAL_*.npz'))

        name_lis = np.stack([os.path.basename(img_name).split('.')[0] for img_name in img_lis])
        semantic_propa_dir = os.path.join(Data_dir, scene_name, f'semantic/{data_mode}', 'deeplab_propa')
        os.makedirs(semantic_propa_dir, exist_ok=True)

        N_img=len(img_lis) 
        for idx in range(N_img):
            img_idx = cv2.imread(img_lis[idx])
            semantic_idx = cv2.imread(semantic_lis[idx], cv2.IMREAD_UNCHANGED)
            girds_idx = cv2.imread(grids_lis[idx], cv2.IMREAD_UNCHANGED)

            valid_score_idx = np.load(valid_score_lis[idx])['arr_0']
            semantic_score_idx = np.load(semantic_score_lis[idx])['arr_0']
            semantic_score_idx = np.exp(-semantic_score_idx)
            semantic_score_idx[~valid_score_idx]=0

            semantic_filter = np.zeros_like(semantic_idx)
            girds_lis = np.unique(girds_idx)
            segs_vis = (np.zeros_like(img_idx)).astype(np.float32)
            for grid in girds_lis:
                if grid == 0:
                    continue
                grid_mask = (girds_idx==grid)
                segs_vis[grid_mask] = np.random.uniform(low=0.1,high=1,size=(3,))  
                semantic_score_mask = semantic_score_idx[grid_mask]
                if semantic_score_mask.sum() == 0:
                    continue
                # 寻找每个mask内可信的语义
                # confidence_mask = (semantic_score_mask>0.7)
                # 只留下最大score的语义
                max_score = semantic_score_mask.max()
                maxscore_mask = (semantic_score_mask>max_score-0.1)

                mask = maxscore_mask

                semantic_mask = semantic_idx[grid_mask]
                semantic_mask[~mask] = 0
                semantic_filter[grid_mask] = semantic_mask
            
            segs_vis = (segs_vis*255).astype(np.uint8)
            cv2.imwrite(f'{semantic_propa_dir}/seg_{name_lis[idx]}.png', segs_vis[...,::-1])

            semantic_filter_vis = colour_map_np[semantic_filter]
            cv2.imwrite(f'{semantic_propa_dir}/{name_lis[idx]}.png', semantic_filter_vis[...,::-1])
            print('1')


            

