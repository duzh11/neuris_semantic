import os, sys
sys.path.append(os.getcwd())
import logging
import cv2
import numpy as np
import random

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

alpha=0.6
random.seed(42)
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(1, 255)
    rgb[1] = random.randint(1, 255)
    rgb[2] = random.randint(1, 255)
    return rgb

from confs.path import lis_name_scenes

data_base='../Data/dataset/indoor'
vis_flag=True
for scene_name in lis_name_scenes:
    logging.info(f'loading instance: {scene_name} ')
    scene_dir=os.path.join(data_base, scene_name)
    for data_mode in ['train', 'test']:
        instance_dir = os.path.join(scene_dir, f'grids/{data_mode}', 'instance')
        ins_postcess_dir = os.path.join(scene_dir, f'grids/{data_mode}', 'instance_post')
        ins_postcess_vis = os.path.join(scene_dir, f'grids/{data_mode}', 'instance_post_vis')
        os.makedirs(ins_postcess_dir, exist_ok=True)
        os.makedirs(ins_postcess_vis, exist_ok=True)

        frame_ids = os.listdir(os.path.join(instance_dir))
        frame_ids =  sorted(frame_ids)

        # ----loading instance---
        instance_list=[]
        for idx in frame_ids:
            file_instance=os.path.join(instance_dir, idx)
            instance = cv2.imread(file_instance, cv2.IMREAD_UNCHANGED)
            instance_list = np.unique(instance)

            # 打碎较大的mask
            max_id = max(instance_list)+1
            for instance_idx in instance_list:
                mask = (instance == instance_idx)
                area = mask.sum()
                if area>40000:    
                    mask_0 = mask[mask]
                    mask_0[:round(area/2)] = False
                    mask[mask] = mask_0
                    instance[mask] = max_id
                    max_id += 1
            
            instance_list = np.unique(instance)
            instance_vis=np.zeros([instance.shape[0], instance.shape[1], 3])
            for instance_idx in instance_list:
                mask = (instance == instance_idx) #分割结果
                if instance_idx==0:
                    instance_vis[mask,:] = np.array([0,0,0])
                    continue
                instance_vis[mask,:] = random_rgb()

            cv2.imwrite(os.path.join(ins_postcess_dir, idx), instance)
            cv2.imwrite(os.path.join(ins_postcess_vis, idx), instance_vis[...,::-1].astype(np.uint8))
