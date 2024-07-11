from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from skimage.color import label2rgb
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import os
import cv2

sam_checkpoint = "/home/du/Proj/2Dv_Semantics/SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam)

img_dir = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
scene_lis = ['scene0378_00']
img_seg = 'SAM'
method = img_seg

for scene_name in tqdm(scene_lis, desc='processing scene...'):
    for data_mode in ['train', 'test']:
        # img
        img_file = sorted(glob(os.path.join(img_dir, scene_name, 'image', data_mode, '*.png')))

        seg_dir = os.path.join(img_dir, scene_name, 'grids', data_mode, method)
        vis_dir = os.path.join(img_dir, scene_name, 'grids', data_mode, method+'_vis')
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        for img_name in tqdm(img_file, desc=f'segging {data_mode}'):
            # SAM RGB/Normal
            img = cv2.imread(img_name)

            # SAM depth
            # depth_raw = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            # depth_mask = (depth_raw != 0)
            # d = np.full(depth_raw.shape, 0, dtype=float)
            # d[depth_mask] = (1 / (depth_raw+1e-6))[depth_mask]
            # colored_depth = (d - np.min(d)) / (np.max(d) - np.min(d))
            # img = mpl.colormaps['viridis'](colored_depth)*255
            # img = img.astype(np.uint8)[:,:,:-1][...,::-1]
            
            # img_vis_dir = os.path.join(img_dir, scene_name, 'grids', data_mode, img_seg)
            # os.makedirs(img_vis_dir, exist_ok=True)
            # cv2.imwrite(os.path.join(img_vis_dir, os.path.basename(img_name)), img)

            masks = mask_generator.generate(img)
            N_seg = len(masks)
            seg = np.zeros((img.shape[0], img.shape[1]))
            segs_vis = (np.zeros_like(img)).astype(np.float32)
            for idx in range(0, N_seg):  
                seg[masks[idx]['segmentation']] = idx+1 #Notice that mask will overlap!!!

            seg_list = np.unique(seg)
            for seg_idx in seg_list:
                mask_idx = (seg==seg_idx)
                if mask_idx.sum()<4000:
                    seg[mask_idx] = 0
                else:
                    segs_vis[mask_idx] = np.random.uniform(low=0.1,high=1,size=(3,))  
            
            segs_vis = (segs_vis*255).astype(np.uint8)

            cv2.imwrite(os.path.join(seg_dir,os.path.basename(img_name)), 
                        seg.astype(np.uint16))
            cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_name)), 
                        segs_vis[...,::-1]) 
            
            





            
        
        