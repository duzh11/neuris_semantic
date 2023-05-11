import os
import logging
import cv2
import numpy as np
import utils.utils_scannet as utils_scannet
import utils.utils_colour as utils_colour

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

semantic_class=40
manhattan=False #Manhattan将door、white board归为wall类，将floor mat归为floor类
# scene_list=['scene0009_01','scene0050_00','scene0084_00','scene0085_00','scene0114_02','scene0580_00','scene0603_00','scene0616_00','scene0617_00','scene0625_00']
scene_list=['scene0009_01','scene0084_00','scene0616_00']

img_h,img_w=480,640
data_base='/home/du/Proj/data/ScanNet'
target_base='/home/du/Proj/NeuRIS/Data/dataset/indoor'

for scene_name in scene_list:
    logging.info(f'loading semantic: {scene_name} ')
    scene_dir=os.path.join(data_base,scene_name)
    target_dir=os.path.join(target_base,scene_name)

    label_filt_dir =  os.path.join(scene_dir, scene_name+'_2d-label-filt')
    semantic_class_dir = label_filt_dir

    frame_ids = os.listdir(os.path.join(target_dir,'image'))
    frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
    frame_ids =  sorted(frame_ids)
    logging.info(f'sample steps: {frame_ids[1]}; Total: {len(frame_ids)}')

    # ----load semantic----
    semantic_list=[]
    for idx in frame_ids:
        file_label=os.path.join(label_filt_dir, 'label-filt', '%d.png'%idx)
        semantic = cv2.imread(file_label, cv2.IMREAD_UNCHANGED)
        semantic = cv2.copyMakeBorder(src=semantic, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)

        if (img_h is not None and img_h != semantic.shape[0]) or \
            (img_w is not None and img_w != semantic.shape[1]):
            semantic = cv2.resize(semantic, (img_w, img_h), interpolation=cv2.INTER_NEAREST)   
        semantic_list.append(semantic)
    semantic_list=np.array(semantic_list)

    # scannet2nyu label
    semantic_nyu_list = semantic_list.copy()

    if semantic_class==3:    
        label_mapping_nyu = utils_scannet.load_scannet_nyu3_mapping(data_base,manhattan=manhattan)
        colour_map_np = utils_colour.nyu3_colour_code
    elif semantic_class==40:
        label_mapping_nyu = utils_scannet.load_scannet_nyu40_mapping(data_base,manhattan=manhattan)
        colour_map_np = utils_colour.nyu40_colour_code
    elif semantic_class==13:
        label_mapping_nyu = utils_scannet.load_scannet_nyu13_mapping(data_base,manhattan=manhattan)
        colour_map_np = utils_colour.nyu13_colour_code
    else:
        label_mapping_nyu = utils_scannet.load_scannet_nyu3_mapping(data_base,manhattan=manhattan)
        colour_map_np = utils_colour.nyu3_colour_code

    for scan_id, nyu_id in label_mapping_nyu.items():
        semantic_nyu_list[semantic_list==scan_id] = nyu_id
    semantic_nyu=np.array(semantic_nyu_list)

    save_semantic=True
    # save_dir=f'semantic_{semantic_class}'
    save_dir=f'semantic_GT'
    # save_vis_dir=f"semantic_{semantic_class}_vis"
    save_vis_dir=f"semantic_GT_vis"

    if save_semantic:
        logging.info(f'save raw label to {os.path.join(target_dir, save_dir)}')
        # save colourised ground truth label to img folder
        label_save_dir = os.path.join(target_dir, save_dir)
        os.makedirs(label_save_dir, exist_ok=True)
        for i in range(len(frame_ids)):
            label = (semantic_nyu[i]).astype(np.uint8)
            cv2.imwrite(os.path.join(label_save_dir, "{}.png".format(frame_ids[i])),label)

    save_semantic_vis=True
    if save_semantic_vis:
        logging.info(f'save color label to {os.path.join(target_dir, save_vis_dir)}')
        # save colourised ground truth label to img folder
        vis_label_save_dir = os.path.join(target_dir, save_vis_dir)
        os.makedirs(vis_label_save_dir, exist_ok=True)
        vis_label = colour_map_np[semantic_nyu]
        for i in range(len(frame_ids)):
            label = vis_label[i].astype(np.uint8)
            cv2.imwrite(os.path.join(vis_label_save_dir, "{}.png".format(frame_ids[i])),label[...,::-1])

logging.info('complete')