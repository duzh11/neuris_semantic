import os
import numpy as np
import logging
import cv2
import trimesh
import plyfile
import open3d as o3d

from glob import glob
from sklearn.metrics import confusion_matrix
import utils.utils_nyu as utils_nyu

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

    if semantic_class==3:
        label=np.array(["object", "wall", "floor"])
    elif semantic_class==40:
        label = np.array(["wall", "floor", "cabinet", "bed", "chair",
                "sofa", "table", "door", "window", "book", 
                "picture", "counter", "blinds", "desk", "shelves",
                "curtain", "dresser", "pillow", "mirror", "floor",
                "clothes", "ceiling", "books", "fridge", "tv",
                "paper", "towel", "shower curtain", "box", "white board",
                "person", "night stand", "toilet", "sink", "lamp",
                "bath tub", "bag", "other struct", "other furntr", "other prop"])
        
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

def evaluate_semantic_2D(dir_scan, 
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
        # merge语义
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

def project_to_mesh(from_mesh, to_mesh, attribute, dist_thresh=None, cmap=None):
    """ Transfers attributs from from_mesh to to_mesh using nearest neighbors

    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.

    Args:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distacne between from and to vertices)

    Returns:
        Trimesh containing transfered attribute
    """

    if len(from_mesh.vertices) == 0:
        to_mesh.vertex_attributes[attribute] = np.zeros((0), dtype=np.uint8)
        to_mesh.visual.vertex_colors = np.zeros((0), dtype=np.uint8)
        return to_mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pred_ids = from_mesh.vertex_attributes[attribute]
    pred_ids[pred_ids == 255] = 0
    
    matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype=np.uint8)
    matched_colors = np.ones((to_mesh.vertices.shape[0], 4), dtype=np.uint8) * 255
    
    for i, vert in enumerate(to_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0]<dist_thresh:
            matched_ids[i] = pred_ids[inds[0]]
            matched_colors[i][:3] = cmap[int(pred_ids[inds[0]])]

    mesh = to_mesh.copy()
    mesh.vertex_attributes['label'] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh

def read_label(fn, is_gt=False):
    a = plyfile.PlyData().read(fn)
    w = np.array(a.elements[0]['label'])
    
    w = w.astype(np.uint8)
    return w

def evaluate_semantic_3D(file_mesh_trgt,
                         file_mesh_pred,
                         file_semseg_pred,
                         semantic_class=40,
                         MANHATTAN=False):
    mesh_trgt = trimesh.load(file_mesh_trgt, process=False)
    mesh_pred = trimesh.load(file_mesh_pred, process=False)
    semseg_pred = np.load(file_semseg_pred)['arr_0']

    vertex_attributes = {}
    vertex_attributes['semseg'] = semseg_pred
    mesh_pred.vertex_attributes = vertex_attributes

    # transfer labels from pred mesh to gt mesh using nearest neighbors
    colour_map_np = utils_nyu.nyu40_colour_code
    mesh_transfer = project_to_mesh(mesh_pred, mesh_trgt, 'semseg', cmap=colour_map_np)
    semseg_pred_trasnfer = mesh_transfer.vertex_attributes['label']

    pred_ids = semseg_pred_trasnfer
    gt_ids = read_label(file_mesh_trgt)

    # merge语义
    semantic_GT = np.array(gt_ids)
    semantic_render = np.array(pred_ids)
    semantic_GT_copy, semantic_render_copy = semantic_GT.copy(), semantic_render.copy()

    if semantic_class==3:
        label_mapping_nyu=mapping_nyu3(manhattan=MANHATTAN)
    if semantic_class==40:
        label_mapping_nyu=mapping_nyu40(manhattan=MANHATTAN)
    for scan_id, nyu_id in label_mapping_nyu.items():
        semantic_GT_copy[semantic_GT==scan_id] = nyu_id
        semantic_render_copy[semantic_render==scan_id] = nyu_id

    if semantic_class>3:
        true_labels=semantic_GT_copy-1
        predicted_labels=semantic_render_copy-1
    else:
        true_labels=semantic_GT_copy
        predicted_labels=semantic_render_copy  

    metric_avg, exsiting_label, class_iou, class_accuray = compute_segmentation_metrics(true_labels=true_labels, 
                                                                                        predicted_labels=predicted_labels, 
                                                                                        semantic_class=semantic_class, 
                                                                                        ignore_label=255)
    
    return mesh_transfer, metric_avg, exsiting_label, class_iou, class_accuray