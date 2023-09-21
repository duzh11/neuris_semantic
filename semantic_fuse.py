# 尝试在前端语义施加单目语义一致性约束
from tqdm import tqdm
from glob import glob
import os
import logging
import cv2
import numpy as np
import random
import utils.utils_colour as utils_colour

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

scene_list=['scene0050_00', 'scene0616_00']
semantic_type_list = ['semantic_pred_repair_0.3', 'semantic_pred_repair_0.5', 'mask2former_repair_0.3', 'mask2former_repair_0.5']
mask_name = ['instance', 'SAM', 'felzenszwalb']
flag_list = ['num', 'mv_similarity']

def fuse_semantic(mask, semantic, mv_similarity,flag='num'):
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
        similarity_mask = mv_similarity[submask]

        # 投票出置信度最高的语义
        semantic_list, counts = np.unique(semantic_mask, return_counts=True)
        
        # 通过数量投票
        if flag == 'num':
            mode_index = np.argmax(counts)
            semantic_maxprob = semantic_list[mode_index]
        elif flag=='mv_similarity':
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
data_base='../Data/dataset/indoor'
colour_map_np = utils_colour.nyu40_colour_code
for scene_name in scene_list:
    logging.info(f'scene: {scene_name}')
    for semantic_type in semantic_type_list:
        logging.info(f'fuse semantic: {semantic_type}')

        data_dir=os.path.join(data_base, scene_name)
        #1. read data
        # semantic
        semantic_dir = os.path.join(data_dir, semantic_type, '*.png')
        semantic_file= glob(semantic_dir)
        semantic_file.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
        # mv_similarity
        mv_similarity_dir = os.path.join(data_dir, 'mv_similarity', semantic_type, '*.npz')
        mv_similarity_file = glob(mv_similarity_dir)
        mv_similarity_file.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
        # instance
        instance_dir = os.path.join(data_dir, 'instance', '*.png')
        instance_file = glob(instance_dir)
        instance_file.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
        # SAM
        SAM_dir = os.path.join(data_dir, 'SAM', '*.png')
        SAM_file = glob(SAM_dir)
        SAM_file.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
        # f
        felzenszwalb_dir = os.path.join(data_dir, 'felzenszwalb_100_1_50_a','*.png')
        felzenszwalb_file = sorted(glob(felzenszwalb_dir))
        
        # 前端语义的单目一致性约束
        save_dir = os.path.join(data_dir, 'sv_fuse', semantic_type)
        save_vis_dir = os.path.join(data_dir, 'sv_fuse_vis',semantic_type)
        

        for idx in tqdm(range(0, len(semantic_file)), desc='semantic fuse'):
            semantic = cv2.imread(semantic_file[idx], cv2.IMREAD_UNCHANGED)
            similarity = np.load(mv_similarity_file[idx])['arr_0']
            similarity[np.isnan(similarity)]=0

            instance = cv2.imread(instance_file[idx], cv2.IMREAD_UNCHANGED)
            SAMseg = cv2.imread(SAM_file[idx], cv2.IMREAD_UNCHANGED)
            felzenszwalb = cv2.imread(felzenszwalb_file[idx], cv2.IMREAD_UNCHANGED)
            seg=[instance, SAMseg, felzenszwalb]
            # fuse semantic
            i=-1
            for mask in mask_name:
                i+=1
                for flag in flag_list:
                    fuse_fir = os.path.join(save_dir, mask+f'_{flag}')
                    fuse_vis_fir = os.path.join(save_vis_dir, mask+f'_{flag}')
                    os.makedirs(fuse_fir, exist_ok=True)
                    os.makedirs(fuse_vis_fir, exist_ok=True)
                    
                    semantic_fuse = fuse_semantic(seg[i], semantic, similarity, flag=flag)
                    semantic_fuse_vis = colour_map_np[semantic_fuse]

                    cv2.imwrite(os.path.join(fuse_fir, '%d.png'%idx), semantic)
                    cv2.imwrite(os.path.join(fuse_vis_fir, '%d.png'%idx), semantic_fuse_vis[...,::-1])
        
        
    