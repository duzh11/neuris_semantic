import os
import logging
import cv2
import numpy as np
import random
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from glob import glob
import utils.utils_nyu as utils_nyu

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

scene_list=['scene0435_02', 'scene0050_00']
semantic_type_list = ['deeplab']
mask_list = ['SAM']
flag_list = ['num']

def fuse_semantic(mask, 
                  semantic, 
                  flag='num', 
                  mv_similarity=None):
    '''
    这个函数用于将semantic和mask进行投票，并返回投票结果
    '''
    semantic_fuse = semantic.copy()
    mask_list = np.unique(mask)
    for i in mask_list:
        if i==0:
            continue
        # 取出子分割块
        submask = (mask==i)
        semantic_mask = semantic_fuse[submask]
        
        # 投票出置信度最高的语义
        semantic_list, counts = np.unique(semantic_mask, return_counts=True)
        
        # 通过数量投票
        if flag == 'num':
            mode_index = np.argmax(counts)
            semantic_maxprob = semantic_list[mode_index]
        elif flag=='mv_similarity':
            similarity_mask = mv_similarity[submask]
            similarity_list=[]
            for semantic_i in semantic_list:
                mask_i = (semantic_mask==semantic_i)
                similarity_i = similarity_mask[mask_i].sum()
                similarity_list.append(similarity_i)

            similarity_counts = np.array(counts)*np.array(similarity_list)
                
            mode_index = np.argmax(similarity_counts)
            semantic_maxprob = semantic_list[mode_index]                

        semantic_fuse[submask] = semantic_maxprob*np.ones_like(semantic_fuse[submask])
    return semantic_fuse

img_h,img_w=480,640
data_root='../Data/dataset/indoor'
colour_map_np = utils_nyu.nyu40_colour_code
for scene_name in scene_list:
    logging.info(f'scene: {scene_name}')
    for data_mode in ['train', 'test']:
        for semantic_type in semantic_type_list:
            logging.info(f'fuse semantic: {semantic_type}')
            data_dir=os.path.join(data_root, scene_name)
            #1. read data
            # semantic
            semantic_dir = os.path.join(data_dir, 'semantic', data_mode, semantic_type, '*.png')
            semantic_file= sorted(glob(semantic_dir))
            # seg mask
            for mask in mask_list:
                mask_dir = os.path.join(data_dir, 'grids', data_mode, mask, '*.png')
                mask_file = sorted(glob(mask_dir))

                for idx in tqdm(range(0, len(semantic_file)), desc='fusing semantic by mask'):
                    semantic = cv2.imread(semantic_file[idx], cv2.IMREAD_UNCHANGED)
                    
                    # fuse semantic
                    seg_mask = cv2.imread(mask_file[idx], cv2.IMREAD_UNCHANGED)  
                    for flag in flag_list:
                        semantic_fuse = fuse_semantic(seg_mask, semantic, flag=flag)
                        semantic_fuse_vis = colour_map_np[semantic_fuse]

                        # save semantic_fuse
                        fuse_dir = os.path.join(data_dir, 'semantic', data_mode, f'{semantic_type}_{mask}_{flag}')
                        fuse_vis_dir = os.path.join(data_dir, 'semantic', data_mode, f'{semantic_type}_{mask}_{flag}_vis')
                        os.makedirs(fuse_dir, exist_ok=True)
                        os.makedirs(fuse_vis_dir, exist_ok=True)

                        cv2.imwrite(os.path.join(fuse_dir, os.path.basename(semantic_file[idx])), semantic_fuse)
                        cv2.imwrite(os.path.join(fuse_vis_dir, os.path.basename(semantic_file[idx])), semantic_fuse_vis[...,::-1])

        
        
    