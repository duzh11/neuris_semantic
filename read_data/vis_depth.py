import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from glob import glob

scene_num='scene0050_00'

depth_dir=os.path.join('../Data/dataset/indoor', scene_num, 'depth/*.png')
depth_file =sorted(glob(depth_dir))

depth_vis = os.path.join('../Data/dataset/indoor', scene_num, 'depth_vis')
os.makedirs(depth_vis, exist_ok=True)

for idx in depth_file:
    gt_depth = (cv2.imread(idx,cv2.IMREAD_UNCHANGED).astype(np.float32))
    gt_depth = gt_depth/20

    gt_depth_map = cv2.convertScaleAbs(gt_depth)
    gt_depth_map_jet = cv2.applyColorMap(gt_depth_map, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(depth_vis, os.path.basename(idx)), gt_depth_map_jet)
