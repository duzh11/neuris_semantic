from skimage.segmentation import slic,felzenszwalb,mark_boundaries
from skimage.color import label2rgb

from glob import glob
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2

alpha=0.8
random.seed(42)
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(1, 255)
    rgb[1] = random.randint(1, 255)
    rgb[2] = random.randint(1, 255)
    return rgb

scene_list=['scene0378_00', 'scene0435_02']
method_name = 'felzenszwalb_100_1_50_a'
delete_small_area = True
for scene in scene_list:
    print(f'---process scene: {scene}---')
    for mode in ['train', 'test']:
        data_dir=f'../Data/dataset/indoor/{scene}'
        img_lis = sorted(glob(os.path.join(data_dir, f'image/{mode}/*.png')))
        seg_dir=os.path.join(data_dir, f'grids/{mode}', method_name)
        vis_dir=os.path.join(data_dir, f'grids/{mode}', f'{method_name}_vis')
        os.makedirs(seg_dir,exist_ok=True)
        os.makedirs(vis_dir,exist_ok=True)

        for img_name in tqdm(img_lis, desc='superpixel segging'):
            img = cv2.imread(img_name)
            
            # segging img
            # segments = slic(img, n_segments=80, compactness=10, sigma=0)
            segs = felzenszwalb(img, scale=100, sigma=1, min_size=50)+1

            # delete small area
            if delete_small_area:
                label_list = np.unique(segs)
                for label in label_list:
                    mask = (segs ==label) #分割结果
                    if mask.sum()<4000:
                        segs[mask]=0
                        continue
            
            # visualizing segments
            segs_vis = label2rgb(segs, img, bg_label=0, alpha=0.5, kind='overlay')
            segs_vis = (segs_vis*255).astype(np.uint8)

            cv2.imwrite(os.path.join(seg_dir, os.path.basename(img_name)), 
                        segs.astype(np.uint16))
            cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_name)), 
                        segs_vis[...,::-1])   
 
    
