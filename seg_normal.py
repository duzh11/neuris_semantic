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
    img_lis = sorted(glob(os.path.join(data_dir, 'edge_100_2_50/*.png')))
    normal_lis = sorted(glob(os.path.join(data_dir, 'pred_normal/*.png')))

    seg_dir=os.path.join(data_dir,'test2')
    vis_dir=os.path.join(data_dir,'test2_vis')
    os.makedirs(seg_dir,exist_ok=True)
    os.makedirs(vis_dir,exist_ok=True)
    
    for idx in range(len(img_lis)):
        img = cv2.imread(img_lis[idx], cv2.IMREAD_ANYDEPTH)
        normal = cv2.imread(normal_lis[idx])

        ### compute PairwiseDistance
        grid_list=np.unique(img)
        for i in grid_list:
            if i==0:
                continue #忽略void类别的semantic consistency
            grid_mask = (img==i)
            normal_mask = normal[grid_mask]
            normal_mask = normal_mask/np.linalg.norm(normal_mask, axis=-1, keepdims=True) #归一化
            normal_mean = np.mean(normal_mask, axis=0)
            normal_diff = np.array(np.dot(normal_mask, normal_mean))
            if np.mean(normal_diff)<0.96:
                img[img==i] = 0
        
        # 上色
        segments = img
        vis=np.zeros_like(normal)
        label_list = np.unique(segments)
        for label in label_list:
            mask = (segments ==label) #分割结果
            if mask.sum()<1000 or label==0:
                segments[mask]=0
                vis[mask,:]=np.array([0,0,0])
                continue
            vis[mask,:] = random_rgb()
        
        # blended_image = cv2.addWeighted(vis, alpha, img, 1-alpha, 0)
        cv2.imwrite(os.path.join(seg_dir, os.path.basename(img_lis[idx])), 
                    segments.astype(np.uint16))
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_lis[idx])), 
                    vis[...,::-1])  






