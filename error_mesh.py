import logging
import numpy as np
from matplotlib import cm
import open3d as o3d

import utils.utils_geometry as GeoUtils

def compute_error_mesh(exp_name, lis_name_scenes, 
                       dir_dataset, name_baseline, dir_results_baseline):
    
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

    color_map = cm.get_cmap('Reds')
    error_bound = 0.01

    for scene_name in lis_name_scenes:
        logging.info(f'\n\nProcess: {scene_name}')
        path_mesh_pred = f'{dir_results_baseline}/{name_baseline}/{scene_name}.ply'
        path_mesh_GT = f'{dir_results_baseline}/GT_refuse/{scene_name}_GT.ply'
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

if __name__=='__main__':
    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    lis_exp_name=['semantic_3_test13']
    lis_name_scenes=['scene0084_00']

    dir_dataset = '../Data/dataset/indoor'
    dir_results_baseline='../exps/evaluation/'
    for exp_name in lis_exp_name:
        logging.info(f'output 3D error mesh of method: {exp_name}')
        name_baseline=f'{exp_name}_refuse'
        compute_error_mesh(exp_name, lis_name_scenes,
                     dir_dataset, name_baseline, 
                     dir_results_baseline)