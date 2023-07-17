from skimage.segmentation import slic, felzenszwalb, mark_boundaries
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torch.nn as nn

alpha=0.6
random.seed(42)
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(1, 255)
    rgb[1] = random.randint(1, 255)
    rgb[2] = random.randint(1, 255)
    return rgb

def normalize(a):
    return (a - a.min())/(a.max() - a.min() + 1e-8)

def compute_pdist(img):
    rgb = (torch.tensor(img)).cuda()
    pdist = nn.PairwiseDistance(p=2)
    rgb_down = pdist(rgb[1:, :, :], rgb[:-1, :, :])
    rgb_right = pdist(rgb[:, 1:, :], rgb[:, :-1, :])
    rgb_down = normalize(rgb_down).cpu().numpy().squeeze()
    rgb_right = normalize(rgb_right).cpu().numpy().squeeze()
    return rgb_down, rgb_right

def detect_edge(img):
    img_mean = cv2.blur(img, (3,3))
    dissimilarity = img-img_mean
    return dissimilarity

scene_list=['scene0050_00', 'scene0426_00', 'scene0616_00']
for scene in scene_list:
    data_dir = f'/home/du/Proj/NeuRIS/Data/dataset/indoor/{scene}'
    img_lis = sorted(glob(os.path.join(data_dir, 'image/*.png')))
    normal_lis = sorted(glob(os.path.join(data_dir, 'pred_normal/*.png')))

    dst_dir=os.path.join(data_dir,'test1_vis')
    seg_dir=os.path.join(data_dir,'test2')
    vis_dir=os.path.join(data_dir,'test2_vis')
    os.makedirs(dst_dir,exist_ok=True)
    os.makedirs(seg_dir,exist_ok=True)
    os.makedirs(vis_dir,exist_ok=True)
    
    for i in range(len(img_lis)):
        img = cv2.imread(img_lis[i])
        normal = cv2.imread(normal_lis[i])

        ### compute PairwiseDistance
        rgb_diss = detect_edge(img)
        normal_diss = detect_edge(normal)

        # dst = np.stack([rgb_diss, normal_diss])
        # dst = np.max(dst, 0)
        alpha=0.5
        dst = alpha*rgb_diss + (1-alpha)*normal_diss
        # render_map = cv2.convertScaleAbs(dst*100/255)
        # render_map_jet = cv2.applyColorMap(render_map, cv2.COLORMAP_JET)
        # cv2.imwrite('./test1.jpg', render_map_jet)
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(img_lis[i])), 
                    dst)

        ## seg
        img = cv2.imread(os.path.join(dst_dir, os.path.basename(img_lis[i])))
        segments = felzenszwalb(img, scale=100, sigma=1, min_size=50)+1
        bounds = mark_boundaries(img,segments)
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

        cv2.imwrite(os.path.join(seg_dir, os.path.basename(img_lis[i])), 
                    segments.astype(np.uint16))
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_lis[i])), 
                    vis[...,::-1])   

