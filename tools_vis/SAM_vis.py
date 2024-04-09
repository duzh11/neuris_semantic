import os, sys
sys.path.append(os.getcwd())

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.color import label2rgb
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import os
import cv2

import utils.utils_io as IOUtils

scene_name = 'scene0050_00'
pic = '2630.png' 
img_dir = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/dataset/indoor'
img_file = os.path.join(img_dir, scene_name, 'image', 'train', pic)

img = cv2.imread(img_file)

### generate segmentation
# SAM
sam_checkpoint = "/home/du/Proj/2Dv_Semantics/SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=64,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100)
masks = mask_generator.generate(img)

N_seg = len(masks)
seg = np.zeros((img.shape[0], img.shape[1]))
for idx in range(0, N_seg):  
    seg[masks[idx]['segmentation']] = idx+1 #Notice that mask will overlap!!!

## SPP
# seg = felzenszwalb(img, scale=100, sigma=1, min_size=50)+1
    
## read from file
# seg_file = os.path.join(img_dir, scene_name, 'grids', 'train', 'SAM', pic)
# seg = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)
    
## clean segs < 4000
seg_list = np.unique(seg)
segs_vis = (np.zeros_like(img)).astype(np.float32)
for seg_idx in seg_list:
    mask_idx = (seg==seg_idx)
    if mask_idx.sum()<1000:
        seg[mask_idx] = 0
    else:
        segs_vis[mask_idx] = np.random.uniform(low=0.1,high=1,size=(3,))  
segs_vis = (segs_vis*255).astype(np.uint8)

## visualize
vis = label2rgb(seg, img, bg_label=0, alpha=0.5, kind='overlay')
# vis = mark_boundaries(vis, seg.astype(int), color=(1, 1, 1), outline_color=(1, 1, 1), background_label=0)
vis = (vis*255)[...,::-1].astype(np.uint8)
# vis = cv2.addWeighted(img, 0.3, segs_vis, 0.7, 0)

vis_file = os.path.join(img_dir, scene_name, 'image', pic)
cv2.imwrite(vis_file, vis)

## fetch specific seg from instance image
# instance_id = 13
# instance_file = os.path.join(img_dir, scene_name, 'grids', 'train', 'instance', pic)
# instance = cv2.imread(instance_file, cv2.IMREAD_UNCHANGED)
# part_mask = instance==instance_id

## fetch specific seg by quering spefici seg_id
# coordinates = [[197, 305], [282, 212], [132, 397]]
# coordinates = [[310, 183], [526, 62], [240, 39], [472, 315], [143, 241]]

# i=0
# for coordinate in coordinates:
#     seg_id = seg[coordinate[1], coordinate[0]]
#     seg_mask = seg==seg_id
#     part_mask = seg_mask

#     part_vis=255*np.ones_like(vis)
#     part_vis[part_mask] = vis[part_mask]
#     instance_vis_file = os.path.join(img_dir, scene_name, 'image', IOUtils.add_file_name_suffix(pic, f'_seg_{i}'))
#     cv2.imwrite(instance_vis_file, part_vis)
#     i+=1