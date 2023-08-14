import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import preprocessing 
scene_num='0616_00'
img_num1='87'
img_num2='0870'
depth_pred='/home/du/Proj/NeuRIS/exps/indoor/neus/scene'+scene_num+'/neuris/depth_npz/'
depth_gt='/home/du/Proj/NeuRIS/Data/dataset/indoor/scene'+scene_num+'/depth/'
pred_name='00160000_'+img_num1+'_reso1.npz'
gt_name=img_num2+'.png'
f1=os.path.join(depth_pred,pred_name)
f2=os.path.join(depth_gt,gt_name)

path_trans_n2w='/home/du/Proj/NeuRIS/Data/dataset/indoor/scene'+scene_num+'/trans_n2w.txt'
scale_mat = np.loadtxt(path_trans_n2w)
scale=scale_mat[0,0]
print(f'scale: {scale}')

render_npz = np.load(f1)['arr_0']
colormap_jet=True
if colormap_jet:
    render_depth=render_npz*scale*50
    gt_depth = (cv.imread(f2,cv.IMREAD_UNCHANGED).astype(np.float32))/20
else:
    render_depth=render_npz
    gt_depth=cv.imread(f2,cv.IMREAD_UNCHANGED)

if colormap_jet:
    gt_depth_map = cv.convertScaleAbs(gt_depth) # 范围阈值(mm)*alpha=255,差不多4米多吧
    gt_depth_map_jet = cv.applyColorMap(gt_depth_map, cv.COLORMAP_JET)
    render_depth_map = cv.convertScaleAbs(render_depth)
    render_depth_map_jet = cv.applyColorMap(render_depth_map, cv.COLORMAP_JET)
else:
    gt_depth_map_jet=(gt_depth / (np.max(gt_depth)+1e-6) * 255).astype(np.uint8)
    render_depth_map_jet=(render_depth / (np.max(render_depth)+1e-6) * 255).astype(np.uint8)

save=True
if save:
    cv.imwrite(depth_pred+'00160000_'+img_num1+'_reso1.png',render_depth_map_jet)
    cv.imwrite(depth_gt+img_num2+'_render.png',gt_depth_map_jet)
print('complete')

