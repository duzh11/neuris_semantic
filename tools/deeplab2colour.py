import os
import logging
import cv2
import numpy as np
import utils.utils_scannet as utils_scannet
import utils.utils_colour as utils_colour

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

img_h,img_w=480,640

target_dir='/home/du/Proj/NeuRIS/Data/dataset/indoor'
semantic_dir='/home/du/Proj/NeuRIS/exps/indoor/neus' 
scene_name='scene0616_00'
method_name='test4_semantic'
target_dir=os.path.join(target_dir,scene_name)
render_dir=os.path.join(semantic_dir,scene_name,method_name)

frame_ids = os.listdir(os.path.join(target_dir,'semantic_deeplab'))
frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
frame_ids =  sorted(frame_ids)
logging.info(f'sample steps: {frame_ids[1]}; Total: {len(frame_ids)}')

# load semantic
deeplab_list=[]
render_list=[]
for idx in frame_ids:
    file_label=os.path.join(target_dir, 'semantic_deeplab', '%d.png'%idx)
    file_render=os.path.join(render_dir, 'semantic_render','00160000_'+'0'*(4-len(str(idx)))+str(idx)+'_reso2.png')
    semantic = cv2.imread(file_label, cv2.IMREAD_UNCHANGED)
    semantic_render=cv2.imread(file_render, cv2.IMREAD_UNCHANGED)
    deeplab_list.append(semantic)
    render_list.append(semantic_render)
deeplab_list=np.array(deeplab_list)
render_list=np.array(render_list)

label_mapping_nyu = utils_scannet.load_scannet_nyu3_mapping('/home/du/Proj/data/ScanNet')
colour_map_np = utils_colour.nyu3_colour_code

save_semantic_vis=True

save_vis_dir="semantic_deeplab_vis"
if False:
    logging.info(f'save color label to {os.path.join(target_dir, save_vis_dir)}')
    # save colourised ground truth label to img folder
    vis_label_save_dir = os.path.join(target_dir, save_vis_dir)
    os.makedirs(vis_label_save_dir, exist_ok=True)
    vis_label = colour_map_np[(deeplab_list/80).astype(np.uint8)]
    for i in range(len(frame_ids)):
        label = vis_label[i].astype(np.uint8)
        cv2.imwrite(os.path.join(vis_label_save_dir, "{}.png".format(frame_ids[i])),label[...,::-1])

save_vis_dir="semantic_render_vis"
if save_semantic_vis:
    logging.info(f'save color label to {os.path.join(render_dir, save_vis_dir)}')
    # save colourised ground truth label to img folder
    vis_label_save_dir = os.path.join(render_dir, save_vis_dir)
    os.makedirs(vis_label_save_dir, exist_ok=True)
    vis_label = colour_map_np[(render_list/80).astype(np.uint8)]
    for i in range(len(frame_ids)):
        label = vis_label[i].astype(np.uint8)
        cv2.imwrite(os.path.join(vis_label_save_dir, "{}.png".format(frame_ids[i])),label[...,::-1])

logging.info('complete')