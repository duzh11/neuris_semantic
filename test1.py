import os
import trimesh
import plyfile

import numpy as np 
import open3d as o3d

import utils.utils_nyu as utils_nyu
import utils.utils_semantic as Semantic_utils

exp_name = 'test/test'
lis_name_scenes = ['scene0378_00']
dir_dataset = '../Data/dataset/indoor'
dir_results_baseline = '../exps/indoor/neus'
semantic_class = 40
colour_map_np = utils_nyu.nyu40_colour_code

def project_to_mesh(from_mesh, to_mesh, attribute, dist_thresh=None):
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
            matched_colors[i][:3] = colour_map_np[int(pred_ids[inds[0]])]

    mesh = to_mesh.copy()
    mesh.vertex_attributes['label'] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh

def read_label(fn, is_gt=False):
    a = plyfile.PlyData().read(fn)
    w = np.array(a.elements[0]['label'])
    
    w = w.astype(np.uint8)
    return w

for scene_name in lis_name_scenes:
    # Read GT Data
    file_mesh_trgt = os.path.join(dir_dataset, f'{scene_name}/{scene_name}_vh_clean_2.labels.ply')
    mesh_trgt = trimesh.load(file_mesh_trgt, process=False)
    # Read pred Data
    file_pred = os.path.join(dir_results_baseline, f'{exp_name}/{scene_name}/meshes')
    file_mesh_pred = os.path.join(file_pred, f'{scene_name}_volume_semantic.ply')
    file_semseg_pred = os.path.join(file_pred, f'{scene_name}_semantic.npz')
    
    mesh_pred = trimesh.load(file_mesh_pred, process=False)
    semseg_pred = np.load(file_semseg_pred)['arr_0']
    
    vertex_attributes = {}
    vertex_attributes['semseg'] = semseg_pred
    mesh_pred.vertex_attributes = vertex_attributes
    # transfer labels from pred mesh to gt mesh using nearest neighbors
    file_mesh_transfer = os.path.join(file_pred, f'{scene_name}_volume_semantic_transfer.ply')
    mesh_transfer = project_to_mesh(mesh_pred, mesh_trgt, 'semseg')
    semseg_pred_trasnfer = mesh_transfer.vertex_attributes['label']
    
    mesh_transfer.export(file_mesh_transfer)

    pred_ids = semseg_pred_trasnfer
    gt_ids = read_label(file_mesh_trgt)

    if semantic_class>3:
        true_labels=np.array(gt_ids)-1
        predicted_labels=np.array(pred_ids)-1
    else:
        true_labels=np.array(gt_ids)
        predicted_labels=np.array(pred_ids)  

    metric_avg, exsiting_label, class_iou, class_accuray = Semantic_utils.compute_segmentation_metrics(true_labels=true_labels, 
                                                                                                        predicted_labels=predicted_labels, 
                                                                                                        semantic_class=semantic_class, 
                                                                                                        ignore_label=255)
    
    print('1')





