from skimage.segmentation import slic,felzenszwalb,mark_boundaries
from skimage.color import label2rgb
from skimage import io
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2

alpha=0.6
random.seed(42)
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(1, 255)
    rgb[1] = random.randint(1, 255)
    rgb[2] = random.randint(1, 255)
    return rgb

scene_list=['scene0050_00', 'scene0426_00', 'scene0616_00']
for scene in scene_list:
    data_dir=f'/home/du/Proj/NeuRIS/Data/dataset/indoor/{scene}'
    img_lis = sorted(glob(os.path.join(data_dir, 'pred_normal/*.png')))
    seg_dir=os.path.join(data_dir,'felzenszwalb_100_1_50_c')
    vis_dir=os.path.join(data_dir,'felzenszwalb_100_1_50_c_vis')
    os.makedirs(seg_dir,exist_ok=True)
    os.makedirs(vis_dir,exist_ok=True)

    for img_name in img_lis:
        img = cv2.imread(img_name)
        # segments = slic(img, n_segments=80, compactness=10, sigma=0)
        segments = felzenszwalb(img, scale=100, sigma=1, min_size=50)+1
        bounds = mark_boundaries(img, segments)
        # 上色
        vis=np.zeros_like(img)
        label_list = np.unique(segments)
        for label in label_list:
            mask = (segments ==label) #分割结果
            if mask.sum()<4000:
                segments[mask]=0
                vis[mask,:]=np.array([0,0,0])
                continue
            vis[mask,:] = random_rgb()
        
        # blended_image = cv2.addWeighted(vis, alpha, img, 1-alpha, 0)
        cv2.imwrite(os.path.join(seg_dir, os.path.basename(img_name)), 
                    segments.astype(np.uint16))
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_name)), 
                    vis[...,::-1])   
 
    
