import numpy as np
import logging
import cv2
from glob import glob
from sklearn.metrics import confusion_matrix

label = np.array(["wall", "floor", "cabinet", "bed", "chair",
        "sofa", "table", "door", "window", "book", 
        "picture", "counter", "blinds", "desk", "shelves",
        "curtain", "dresser", "pillow", "mirror", "floor",
        "clothes", "ceiling", "books", "fridge", "tv",
        "paper", "towel", "shower curtain", "box", "white board",
        "person", "night stand", "toilet", "sink", "lamp",
        "bath tub", "bag", "other struct", "other furntr", "other prop"])

def mapping_nyu3(manhattan=False):
    mapping = {}
    for i in range(41):
        if i in [0, 1, 2]:
            mapping[i]=i
        else:
            mapping[i]=0
        if manhattan:
            if i==8: # regard door as wall
                mapping[i]=1
            elif i == 30: # regard white board as wall
                mapping[i]=1
            elif i == 20: # regard floor mat as floor
                mapping[i]=2
    return mapping

def mapping_nyu40(manhattan=False):
    mapping = {}
    for i in range(41):
        mapping[i]=i
        if manhattan:
            if i==8: # regard door as wall
                mapping[i]=1
            elif i == 30: # regard white board as wall
                mapping[i]=1
            elif i == 20: # regard floor mat as floor
                mapping[i]=2
    return mapping

def compute_segmentation_metrics(true_labels, predicted_labels, semantic_class, ignore_label):

    true_labels=np.array(true_labels)
    predicted_labels=np.array(predicted_labels)
    
    if (true_labels == ignore_label).all():
        return [0]*4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    valid_pix_ids = true_labels!=ignore_label
    predicted_labels = predicted_labels[valid_pix_ids] 
    true_labels = true_labels[valid_pix_ids]
    
    # 利用confusion matrix进行计算
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(0, semantic_class)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    exsiting_label = label[exsiting_class_mask]
    # ACC
    average_accuracy = np.mean(np.diagonal(norm_conf_mat)[exsiting_class_mask]) #平均精度
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)) #总精度
    class_accuray_0=np.diagonal(norm_conf_mat).copy() #类别精度
    class_accuray=class_accuray_0[exsiting_class_mask]
    
    # IoU
    class_iou_0 = np.zeros(semantic_class)
    for class_id in range(semantic_class):
        class_iou_0[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id])) 
    
    class_iou = class_iou_0[exsiting_class_mask]
    average_iou = np.mean(class_iou) #平均IoU
    freq = conf_mat.sum(axis=1) / conf_mat.sum()
    FW_iou = (freq[exsiting_class_mask] * class_iou_0[exsiting_class_mask]).sum()

    metric_avg = [average_accuracy, total_accuracy, average_iou, FW_iou]
    return metric_avg, exsiting_label, class_iou, class_accuray

def evaluate_semantic(dir_scan, 
                      dir_exp,
                      acc,
                      data_mode,
                      semantic_class=40,
                      iter=160000,
                      MANHATTAN=False):    
    # loading data
    semantic_GT_lis = sorted(glob(f'{dir_scan}/semantic/{data_mode}/semantic_GT/*.png'))
    semantic_render_lis = sorted(glob(f'{dir_exp}/semantic/{data_mode}/{acc}/{iter:0>8d}_*.npz'))

    semantic_GT_list=[]
    semantic_render_list=[]
    for idx in range(len(semantic_GT_lis)):
        semantic_GT=(cv2.imread(semantic_GT_lis[idx], cv2.IMREAD_UNCHANGED)).astype(np.uint8)
        semantic_render=(np.load(semantic_render_lis[idx])['arr_0']).astype(np.uint8)

        # 考虑渲染的语义是(320,240)
        reso=semantic_GT.shape[0]/semantic_render.shape[0]
        if reso>1:
            semantic_GT=cv2.resize(semantic_GT, (semantic_render.shape[1],semantic_render.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        semantic_GT_copy = semantic_GT.copy()
        semantic_render_copy = semantic_render.copy()
        # 遵循Manhattan-sdf的语义merge策略
        if MANHATTAN:
            if semantic_class==3:
                label_mapping_nyu=mapping_nyu3(manhattan=MANHATTAN)
            if semantic_class==40:
                label_mapping_nyu=mapping_nyu40(manhattan=MANHATTAN)
            for scan_id, nyu_id in label_mapping_nyu.items():
                semantic_GT_copy[semantic_GT==scan_id] = nyu_id
                semantic_render_copy[semantic_render==scan_id] = nyu_id
        
        semantic_GT_list.append(np.array(semantic_GT_copy))
        semantic_render_list.append(semantic_render_copy)

    if semantic_class>3:
        true_labels=np.array(semantic_GT_list)-1
        predicted_labels=np.array(semantic_render_list)-1
    else:
        true_labels=np.array(semantic_GT_list)
        predicted_labels=np.array(semantic_render_list)  

    metric_avg, exsiting_label, class_iou, class_accuray = compute_segmentation_metrics(true_labels=true_labels, 
                                                                                        predicted_labels=predicted_labels, 
                                                                                        semantic_class=semantic_class, 
                                                                                        ignore_label=255)
    
    logging.info(f'exsiting_label: {exsiting_label}')
    return metric_avg, exsiting_label, class_iou, class_accuray

