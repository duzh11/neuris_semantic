import os, sys
sys.path.append(os.getcwd())
import torch
import cv2

from glob import glob
from tqdm import tqdm

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import utils.utils_nyu as NyuUtils
colour_map_np = NyuUtils.nyu40_colour_code

import uncertrainty.compute_viewAL as compute_viewAL
        
from confs.path import lis_name_scenes
# lis_name_scenes=['scene0378_00','scene0435_02'] 
Data_dir = '../Data/dataset/indoor'
lis_semantic_type = ['deeplab']
uncertainty_type=['viewAL']

for scene_name in lis_name_scenes:
        print(f'\n\n---processing scene_name :{scene_name}----\n')
        for semantic_type in lis_semantic_type:
                print(f'---processing semantics :{semantic_type}---\n')
                for data_mode in ['train', 'test']:
                        ## Reading data
                        intrinsic_dir = os.path.join(Data_dir, scene_name, 'intrinsic_color_crop1248_resize640.txt')
                        pose_dir = os.path.join(Data_dir, scene_name, f'pose/{data_mode}')
                        img_dir = os.path.join(Data_dir, scene_name, f'image/{data_mode}')
                        depth_dir = os.path.join(Data_dir, scene_name, f'depth/{data_mode}')
                        semantic_dir = os.path.join(Data_dir, scene_name, f'semantic/{data_mode}', semantic_type)
                        logits_dir = os.path.join(Data_dir, scene_name, f'semantic/{data_mode}', f'{semantic_type}_logits')

                        pose_lis = sorted(glob(f'{pose_dir}/*.txt'))
                        img_lis = sorted(glob(f'{img_dir}/*.png'))
                        depth_lis = sorted(glob(f'{depth_dir}/*.png'))
                        semantic_lis = sorted(glob(f'{semantic_dir}/*.png'))
                        logits_lis = sorted(glob(f'{logits_dir}/*.npz'))

                        intrinsic_np = np.loadtxt(intrinsic_dir)
                        name_lis = np.stack([os.path.basename(img_name).split('.')[0] for img_name in img_lis])
                        pose_np = np.stack([np.loadtxt(pose_name) for pose_name in pose_lis])
                        img_np = np.stack([cv2.imread(img_name) for img_name in img_lis])
                        depth_np = np.stack([cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)/1000 for depth_name in depth_lis])
                        semantic_np = np.stack([cv2.imread(seg_name, cv2.IMREAD_UNCHANGED).astype(np.float32) for seg_name in semantic_lis])
                        
                        N_img, H, W = semantic_np.shape

                        ## 2.compute semantic uncertainty
                        print('compute uncertainty...')
                         ### 2.1 sv_entropy
                        entropy_dir = os.path.join(Data_dir, scene_name, f'semantic_uncertainty/{data_mode}', semantic_type, 'entropy')
                        entropy_vis_dir = entropy_dir+'_vis'
                        os.makedirs(entropy_dir, exist_ok=True)
                        os.makedirs(entropy_vis_dir, exist_ok=True)
                        for idx in tqdm(range(N_img), desc='computing entropy'):
                                logits_idx = np.load(logits_lis[idx])['arr_0']
                                prob_idx = scipy.special.softmax(logits_idx, axis=-1)
                                ### computing entropy
                                prob_2_uncertainty = lambda x: np.sum(-np.log2(x+1e-12)*x, axis=-1)
                                entropy = prob_2_uncertainty(prob_idx)
                                np.savez(os.path.join(entropy_dir, 'entropy_'+name_lis[idx]+'.npz'), entropy)
                                ### vis entropy
                                colormap_func = matplotlib.cm.get_cmap("jet")
                                entropy_vis = colormap_func(entropy)[:, :, :3]
                                entropy_vis = (entropy_vis[...,::-1]*255).astype('uint8')

                                entropy_score_vis = colormap_func(np.exp(-entropy))[:, :, :3]
                                entropy_score_vis = (entropy_score_vis*255).astype('uint8')
                                
                                img_cat=(255 * np.ones((H, 10, 3))).astype('uint8')
                                lis = [img_np[idx], img_cat, colour_map_np[semantic_np[idx].astype(int)][...,::-1], img_cat, \
                                       entropy_vis, img_cat, entropy_score_vis]
                                lis = np.concatenate(lis, axis=1)
                                cv2.imwrite(os.path.join(entropy_vis_dir, name_lis[idx]+'.png'), lis)

                        ### 2.2 viewAL
                        viewAL_dir = os.path.join(Data_dir, scene_name, f'semantic_uncertainty/{data_mode}', semantic_type, 'viewALNeighbour')
                        os.makedirs(viewAL_dir, exist_ok=True)

                        compute_viewAL.viewAL_uncertainty(name_lis, 
                                                          intrinsic_np, 
                                                          img_np, 
                                                          pose_np, 
                                                          depth_np, 
                                                          semantic_np, 
                                                          viewAL_dir,
                                                          logits_lis = logits_lis,
                                                          validate_flag = False)