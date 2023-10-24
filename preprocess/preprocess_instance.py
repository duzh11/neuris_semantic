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

img_h,img_w=480,640
data_base='/home/du/Proj/Dataset/ScanNet/scans'
target_base='../Data/dataset/indoor'
vis_flag=True
for scene_name in lis_name_scenes:
    logging.info(f'loading instance: {scene_name} ')
    scene_dir=os.path.join(data_base, scene_name)
    target_dir=os.path.join(target_base, scene_name)
    for data_mode in ['train', 'test']:
        instance_filt_dir =  os.path.join(scene_dir, 'instance-filt')
        target_instance_dir = os.path.join(target_dir, f'grids/{data_mode}', 'instance')
        target_vis_dir = os.path.join(target_dir, f'grids/{data_mode}', 'instance_vis')
        os.makedirs(target_instance_dir, exist_ok=True)
        os.makedirs(target_vis_dir, exist_ok=True)

        frame_ids = os.listdir(os.path.join(target_dir, f'image/{data_mode}'))
        frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
        frame_ids =  sorted(frame_ids)
        logging.info(f'sample steps: {int(frame_ids[1])-int(frame_ids[0])}; Total: {len(frame_ids)}')

        # ----loading instance---
        instance_list=[]
        for idx in frame_ids:
            file_instance=os.path.join(instance_filt_dir, '%d.png'%idx)
            instance = cv2.imread(file_instance, cv2.IMREAD_UNCHANGED)

            if (img_h is not None and img_h != instance.shape[0]) or \
                (img_w is not None and img_w != instance.shape[1]):
                instance_crop = instance[16:instance.shape[0]-16, 24:instance.shape[1]-24]
                instance = cv2.resize(instance_crop, (img_w, img_h), interpolation=cv2.INTER_NEAREST)   
            
            cv2.imwrite(os.path.join(target_instance_dir, f"{idx:04d}.png"), instance)
            instance_list.append(instance)
        instance_list=np.array(instance_list)
        
        # ----visualize instance---
        if vis_flag:
            instance_vis=np.zeros([len(instance_list), img_h, img_w, 3])
            label_list = np.unique(instance_list)
            for label in label_list:
                mask = (instance_list ==label) #分割结果
                if label==0:
                    instance_vis[mask,:] = np.array([0,0,0])
                    continue
                instance_vis[mask,:] = random_rgb()

            for idx in range(len(instance_list)):
                vis = instance_vis[idx]
                cv2.imwrite(os.path.join(target_vis_dir, f"{frame_ids[idx]:04d}.png"), vis[...,::-1])