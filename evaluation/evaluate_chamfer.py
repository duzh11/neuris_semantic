import torch
import numpy as np
import open3d as o3d
import os as os
from datetime import datetime
import logging

import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils
import utils.utils_nyu as utils_nyu
import evaluation.EvalScanNet as EvalScanNet

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return np.array(indices), np.array(distances)

def compute_chamfer(pcd_pred, pcd_gt, colour_map_np, draw_label=True, Manhattan=True):
    verts_pred = np.asarray(pcd_pred.points)
    verts_gt = np.asarray(pcd_gt.points)
    colors_gt =np.asarray(pcd_gt.colors)
    labels_gt=np.zeros(colors_gt.shape[0])
    
    #标签索引
    for idx in range(colour_map_np.shape[0]):
        for jdx in range(colors_gt.shape[0]):
            if np.linalg.norm(colors_gt[jdx]-colour_map_np[idx]/255)<0.05:
                labels_gt[jdx]=int(idx)
    #检查标签是否正确
    if draw_label:
        verts_colors=colour_map_np[labels_gt.squeeze().astype(np.uint8)]/255
        pcd_gt.colors=o3d.utility.Vector3dVector(verts_colors)
        o3d.visualization.draw_geometries([pcd_gt])

    indices_t, dist_t = nn_correspondance(verts_pred, verts_gt)  # gt->pred
    indices_p, dist_p = nn_correspondance(verts_gt, verts_pred)  # pred->gt
    dist_t=dist_t**2
    dist_p=dist_p**2
    #gt->pred
    chamfer=np.mean(dist_t)
    if Manhattan:
        indices_gt_wall= (labels_gt==1) | (labels_gt==8) | (labels_gt==30)
        indices_gt_floor= (labels_gt==2) | (labels_gt==20)
    else:
        indices_gt_wall= (labels_gt==1) 
        indices_gt_floor= (labels_gt==2)       
    indices_gt_other= (~indices_gt_wall & ~indices_gt_floor)
    assert indices_gt_wall.sum()+indices_gt_floor.sum()+indices_gt_other.sum()==dist_t.shape[0]
    
    chamfel_wall=np.mean(dist_t[indices_gt_wall])
    chamfel_floor=np.mean(dist_t[indices_gt_floor])
    chamfel_other=np.mean(dist_t[indices_gt_other])
    #pred->gt
    indices_pred_wall= (labels_gt[indices_p]==1) 
    indices_pred_floor= (labels_gt[indices_p]==2)
    indices_pred_other= (~indices_pred_wall & ~indices_pred_floor)
    assert indices_pred_wall.sum()+indices_pred_floor.sum()+indices_pred_other.sum()==dist_p.shape[0]
    
    chamfer+=np.mean(dist_p)
    chamfel_wall+=np.mean(dist_p[indices_pred_wall])
    chamfel_floor+=np.mean(dist_p[indices_pred_floor])
    chamfel_other+=np.mean(dist_p[indices_pred_other])
    return np.array([chamfer,chamfel_wall,chamfel_floor,chamfel_other])

def evaluate_chamfer(exp_name,lis_name_scenes, name_baseline, dir_results_baseline, down_sample=0.02):
    metrics_eval_all=[]
    colour_map_np = utils_nyu.nyu40_colour_code    
    
    for scene_name in lis_name_scenes:
        logging.info(f'begin: {scene_name}')
        file_pred = f'{dir_results_baseline}/{name_baseline}/{scene_name}.ply'
        file_gt=f'{dir_results_baseline}/GT_colors/{scene_name}.ply'

        pcd_pred = GeoUtils.read_point_cloud(file_pred)
        pcd_gt = GeoUtils.read_point_cloud(file_gt)

        if down_sample:
            pcd_pred = pcd_pred.voxel_down_sample(down_sample)
            pcd_gt = pcd_gt.voxel_down_sample(down_sample)

        metrics_eval=compute_chamfer(pcd_pred, pcd_gt, colour_map_np)
        logging.info(f'{metrics_eval}')
        metrics_eval_all.append(metrics_eval*100)
    
    path_log = f'{dir_results_baseline}/{name_baseline}/eval_{name_baseline}_Chamfer_markdown.txt'
    markdown_header=f'| scene_name   |    Method|    All|    Wall|  Floor|  Other|\n'
    markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- |\n'
    EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                    header = markdown_header, 
                                                    exp_name=exp_name,
                                                    results = metrics_eval_all, 
                                                    names_item = lis_name_scenes, 
                                                    save_mean = True, 
                                                    mode = 'w')

if __name__=='__main__':
    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    lis_exp_name=['semantic_3_test13']
    lis_name_scenes=['scene0084_00']
    for exp_name in lis_exp_name:
        logging.info(f'compute chamfer of method: {exp_name}')
        name_baseline=f'{exp_name}_refuse'
        dir_results_baseline='../exps/evaluation'
        evaluate_chamfer(exp_name,lis_name_scenes,
                         name_baseline,
                         dir_results_baseline)
