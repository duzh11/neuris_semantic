# 从GT semantic和GT instance中抽取mask
# 只在wall和floor上面考虑，不同方向的墙壁具有不同的id
import os, sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from skimage.color import label2rgb

alpha=0.8
random.seed(42)
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(1, 255)
    rgb[1] = random.randint(1, 255)
    rgb[2] = random.randint(1, 255)
    return rgb

from confs.path import lis_name_scenes

WALL_ID = 1
FLOOR_ID = 2
for scene in lis_name_scenes:
    print(f'---process scene: {scene}---')
    for mode in ['train', 'test']:
        data_dir=f'../Data/dataset/indoor/{scene}'
        img_file_lis = sorted(glob(os.path.join(data_dir, f'image/{mode}/*.png')))
        instance_file_lis = sorted(glob(os.path.join(data_dir, f'grids/{mode}/instance/*.png')))
        semantic_file_lis = sorted(glob(os.path.join(data_dir, f'semantic/{mode}/semantic_GT/*.png')))

        plane_dir=os.path.join(data_dir, f'plane/{mode}', 'GT_semseg')
        plane_vis_dir=os.path.join(data_dir, f'plane/{mode}', 'GT_semseg_vis')
        os.makedirs(plane_dir,exist_ok=True)
        os.makedirs(plane_vis_dir,exist_ok=True)

        N = len(img_file_lis)
        for idx in tqdm(range(N), desc='Obtaining GT plane of wall and floor...'):
            img = cv2.imread(img_file_lis[idx])
            instance = cv2.imread(instance_file_lis[idx], cv2.IMREAD_UNCHANGED)
            semantic = cv2.imread(semantic_file_lis[idx], cv2.IMREAD_UNCHANGED)
            # 获取GT 的WALL和floor mask
            plane = np.zeros_like(semantic)
            wall_mask = semantic==WALL_ID
            floor_mask = semantic==FLOOR_ID
            plane[floor_mask] = 1
            # 由于wall具有不同方向
            wall_instance_id = np.unique(instance[wall_mask])
            plane_id = 2
            for wall_id in wall_instance_id:
                wall_id_mask = instance==wall_id
                plane[wall_id_mask] = plane_id
                plane_id +=1
            # 
            plane_vis = label2rgb(plane, img, bg_label=0, alpha=0.5, kind='overlay')
            plane_vis = (plane_vis*255).astype(np.uint8)

            cv2.imwrite(os.path.join(plane_dir, os.path.basename(img_file_lis[idx])), 
                        plane.astype(np.uint8))
            cv2.imwrite(os.path.join(plane_vis_dir, os.path.basename(img_file_lis[idx])), 
                        plane_vis[...,::-1])   
            





            
        