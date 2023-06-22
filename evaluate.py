import os, logging
from datetime import datetime
import numpy as np
import cv2
from matplotlib import cm
import open3d as o3d

import evaluation.EvalScanNet as EvalScanNet
import utils.utils_semantic as SemanticUtils
import utils.utils_mesh as MeshUtils
import utils.utils_geometry as GeoUtils
import utils.utils_chamfer as ChamferUtils

def nn_correspondance(verts1, verts2):
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
    return indices, distances

def evalute_mesh(exp_name, 
                 lis_name_scenes, 
                 name_baseline,
                 dir_dataset='../Data/dataset/indoor',  
                 dir_results_baseline='../exps/evaluation',
                 eval_threshold=[0.02,0.05,0.10]):

    metrics_eval_mesh = []
    for scene_name in lis_name_scenes:
        
        logging.info(f'\n\nProcess mesh: {scene_name}')
        scene_dir=os.path.join(dir_dataset,scene_name)
        output_dir=f'{dir_results_baseline}/{name_baseline}'
        os.makedirs(output_dir,exist_ok=True)
        logging.info(f'output_dir:{output_dir}')
        
        path_mesh_pred = f'{dir_results_baseline}/{exp_name}/{scene_name}.ply'
        path_mesh_GT=os.path.join(scene_dir,f'{scene_name}_vh_clean_2.ply')
        mesh_pred_dir, mesh_GT_dir= MeshUtils.refuse_mesh(scene_name=scene_name,
                                                              scene_dir=scene_dir,
                                                              output_dir=output_dir,
                                                              path_mesh_pred=path_mesh_pred,
                                                              path_mesh_GT=path_mesh_GT)
        metrices_eval=[]
        for ii in eval_threshold:
            metrices = EvalScanNet.evaluate_geometry_neucon(mesh_pred_dir, mesh_GT_dir, 
                                                threshold=ii, down_sample=.02)
            metrices_eval.append(metrices[-2])

        metrices_eval.append(metrices[-1])
        logging.info(metrices_eval)
        metrics_eval_mesh.append(metrices_eval)
    
    return metrics_eval_mesh    

def Error_mesh(lis_name_scenes,
                name_baseline,  
                dir_results_baseline='../exps/evaluation'):

    color_map = cm.get_cmap('Reds')
    error_bound = 0.01
    gt_dir=os.path.dirname(dir_results_baseline)

    for scene_name in lis_name_scenes:
        logging.info(f'\n\nProcess: {scene_name}')
        path_mesh_pred = f'{dir_results_baseline}/{name_baseline}/{scene_name}.ply'
        path_mesh_GT = f'{gt_dir}/GT_refuse/{scene_name}_GT.ply'
        # path_mesh_pred = f'{dir_results_baseline}/{name_baseline}/{scene_name}_clean_bbox_faces_mask.ply'#ply
        # path_mesh_GT = f'{dir_dataset}/{scene_name}/{scene_name}_vh_clean_2.ply'
        
        error_mesh=f'{dir_results_baseline}/{name_baseline}/{scene_name}_error_{error_bound}.ply'

        mesh_gt = GeoUtils.read_triangle_mesh(path_mesh_GT)
        verts_gt = np.asarray(mesh_gt.vertices)
        triangles_gt = np.asarray(mesh_gt.triangles)

        mesh_pred = GeoUtils.read_triangle_mesh(path_mesh_pred)
        verts_pred = np.asarray(mesh_pred.vertices)
        triangles_pred = np.asarray(mesh_pred.triangles)

        indices_a, dist_a = nn_correspondance(verts_pred, verts_gt)
        dist_a = np.array(dist_a)
        
        indices_r, dist_r = nn_correspondance(verts_gt, verts_pred)
        dist_r = np.array(dist_r)
        
        dist_a1=np.array(dist_r)
        dist_a1[indices_a]=dist_a

        dist=dist_r**2+dist_a1**2

        dist_score = dist.clip(0, error_bound) / error_bound
        colors = color_map(dist_score)[:, :3]

        GeoUtils.save_mesh(error_mesh, verts_pred, triangles_pred, colors)    

def label_mesh(exp_name,
               lis_name_scenes,
               name_baseline,
               dir_dataset='../Data/dataset/indoor',
               dir_results_baseline='../exps/evaluation'):
    for scene_name in lis_name_scenes:
        scene_dir=os.path.join(dir_dataset,scene_name)
        output_dir=f'{dir_results_baseline}/{name_baseline}'
        path_mesh_pred=f'{dir_results_baseline}/{exp_name}/{scene_name}.ply'
        MeshUtils.label3D(exp_name=exp_name,
                scene_name=scene_name,
                scene_dir=scene_dir,
                output_dir=output_dir,
                path_mesh_pred=path_mesh_pred)
    

def save_result(name_baseline,
                lis_name_scenes,
                metrics_eval_semantic=None,
                metrics_iou=None,
                metrics_eval_mesh=None,
                metrics_eval_chamfer=None,
                dir_results_baseline='../exps/evaluation'):
    ### save semantic result
    str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path_log = f'{dir_results_baseline}/{exp_name}_refuse/eval_{exp_name}_{str_date}_markdown.txt'

    if np.sum(metrics_eval_semantic)>0.01:
        markdown_header='Eval metrics\n| scene_ name   |   Method|  Acc.|  M_Acc|  M_IoU| FW_IoU|\n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        exp_name=exp_name,
                                                        results = metrics_eval_semantic, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'w')
    if np.sum(metrics_eval_semantic)>0.01:
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = '\nIoU\n', 
                                                        exp_name=exp_name,
                                                        results = metrics_iou, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = False, 
                                                        mode = 'a')

    ### save 3D result
    if np.sum(metrics_eval_mesh)>0.01:
        markdown_header='\nEval mesh\n| scene_name   |    Method| F-score$_{0.03}$| F-score$_{0.05}$| F-score$_{0.07}$| Chamfer|\n'
        markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        exp_name=exp_name,
                                                        results = metrics_eval_mesh, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'a')
    if np.sum(metrics_eval_chamfer)>0.01:
        ### save chamfer result
        markdown_header=f'\nEval chamfer\n| scene_name   |    Method|    All|    Wall|  Floor|  Other|\n'
        markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        exp_name=exp_name,
                                                        results = metrics_eval_chamfer, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'a')

if __name__=='__main__':
    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)    
    # lis_exp_name=[f'semantic_40_test{i}' for i in range(1,13)]
    method='neus'
    lis_exp_name=['neus_ablation_test8']
    # lis_name_scenes=['scene0015_00','scene0025_00','scene0169_00','scene0414_00','scene0426_00','scene0568_00']
    lis_name_scenes=['scene0616_00']
    numclass=40
    eval_threshold=[0.03,0.05,0.07]
    
    dir_dataset='../Data/dataset/indoor'
    exp_dir='../exps/indoor/neus'
    dir_results_baseline = f'../exps/evaluation/{method}'

    metrics_eval_semantic=[]
    metrics_eval_mesh=[]
    metrics_eval_chamfer=[]
    metrics_iou=[]

    for exp_name in lis_exp_name:
        name_baseline=f'{exp_name}_refuse'
        # logging.info(f'------Evaluate semantics: {exp_name}')
        # metrics_eval_semantic, metrics_acc, metrics_iou=SemanticUtils.evaluate_semantic(exp_name, 
        #                                                           lis_name_scenes,
        #                                                           numclass)
        
        # label_mesh(exp_name, lis_name_scenes, name_baseline, dir_results_baseline=dir_results_baseline)
    
        logging.info(f'------Evaluate mesh: {exp_name}')
        metrics_eval_mesh=evalute_mesh(exp_name, 
                                       lis_name_scenes, 
                                       name_baseline,
                                       eval_threshold=eval_threshold,
                                       dir_results_baseline=dir_results_baseline)

        logging.info(f'------Evaluate chamfer distance: {exp_name}')
        metrics_eval_chamfer=ChamferUtils.evaluate_chamfer(lis_name_scenes, 
                                                           name_baseline,
                                                           dir_results_baseline=dir_results_baseline)

        logging.info(f'------Evaluate 3D error mesh: {exp_name}')
        Error_mesh(lis_name_scenes, name_baseline, dir_results_baseline=dir_results_baseline)

        save_result(name_baseline,lis_name_scenes,
                    metrics_eval_semantic=metrics_eval_semantic,
                    metrics_iou=metrics_iou,
                    metrics_eval_mesh=metrics_eval_mesh,
                    metrics_eval_chamfer=metrics_eval_chamfer,
                    dir_results_baseline=dir_results_baseline)



        



