import os
import logging
import cv2
import numpy as np
import random
import shutil
import utils.utils_scannet as utils_scannet
import utils.utils_colour as utils_colour

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

scene_list=['scene0378_00', 'scene0435_02', 'scene0648_00', 'scene0474_01', 'scene0030_00']

img_h,img_w=480,640
data_base='/home/du/Proj/Dataset/ScanNet/scans'
target_base='../Data/dataset/indoor'
vis_flag=True
for scene_name in scene_list:
    logging.info(f'loading instance: {scene_name} ')
    scene_dir=os.path.join(data_base, scene_name)
    target_dir=os.path.join(target_base, scene_name)

    instance_filt_dir =  os.path.join(scene_dir, 'instance-filt')
    target_instance_dir = os.path.join(target_dir, 'grids', 'instance')
    target_vis_dir = os.path.join(target_dir, 'grids', 'instance_vis')
    os.makedirs(target_instance_dir, exist_ok=True)
    os.makedirs(target_vis_dir, exist_ok=True)

    frame_ids = os.listdir(os.path.join(target_dir, 'image'))
    frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
    frame_ids =  sorted(frame_ids)
    logging.info(f'sample steps: {frame_ids[1]}; Total: {len(frame_ids)}')

    # ----load instance---
    instance_list=[]
    for idx in frame_ids:
        file_instance=os.path.join(instance_filt_dir, '%d.png'%idx)
        instance = cv2.imread(file_instance, cv2.IMREAD_UNCHANGED)
        # instance = cv2.copyMakeBorder(src=instance, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)

        if (img_h is not None and img_h != instance.shape[0]) or \
            (img_w is not None and img_w != instance.shape[1]):
            instance = cv2.resize(instance, (img_w, img_h), interpolation=cv2.INTER_NEAREST)   
        
        cv2.imwrite(os.path.join(target_instance_dir, "{}.png".format(idx)), instance)
        instance_list.append(instance)
    instance_list=np.array(instance_list)
    
    if vis_flag:
        instance_vis=np.zeros([len(instance_list), img_h, img_w, 3])
        label_list = np.unique(instance_list)
        for label in label_list:
            mask = (instance_list ==label) #分割结果
            if label==0:
                instance_vis[mask,:] = np.array([0,0,0])
                continue
            instance_vis[mask,:] = random_rgb()
    
        # blended_image = cv2.addWeighted(vis, alpha, img, 1-alpha, 0)
        for i in range(len(instance_list)):
            vis = instance_vis[i]
            cv2.imwrite(os.path.join(target_vis_dir, "{}.png".format(frame_ids[i])), 
                        vis[...,::-1])