# borrowed from nerfingmvs and neuralreon

import os, cv2,logging
import numpy as np
import open3d as o3d
from matplotlib import cm
from tqdm import tqdm

import utils.utils_geometry as GeoUtils
import utils.utils_TSDF as TSDFUtils
import utils.utils_io as IOUtils
import utils.utils_nyu as NYUUtils

def load_gt_depths(image_list, datadir, H=None, W=None):
    depths = []
    masks = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{:04d}.png'.format(int(frame_id)))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000
        
        if H is not None:
            mask = (depth > 0).astype(np.uint8)
            depth_resize = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            depths.append(depth_resize)
            masks.append(mask_resize > 0.5)
        else:
            depths.append(depth)
            masks.append(depth > 0)
    return np.stack(depths), np.stack(masks)

def load_depths_npy(image_list, datadir, H=None, W=None):
    depths = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{}_depth.npy'.format(frame_id))
        if not os.path.exists(depth_path):
            depth_path = os.path.join(datadir, '{}.npy'.format(frame_id))
        depth = np.load(depth_path)
        
        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            depths.append(depth_resize)
        else:
            depths.append(depth)

    return np.stack(depths)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def depth_evaluation(gt_depths, pred_depths, savedir=None, pred_masks=None, min_depth=0.1, max_depth=20, scale_depth = False):
    assert gt_depths.shape[0] == pred_depths.shape[0]

    gt_depths_valid = []
    pred_depths_valid = []
    errors = []
    num = gt_depths.shape[0]
    for i in range(num):
        gt_depth = gt_depths[i]
        mask = (gt_depth > min_depth) * (gt_depth < max_depth)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = cv2.resize(pred_depths[i], (gt_width, gt_height))

        if pred_masks is not None:
            pred_mask = pred_masks[i]
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_width, gt_height)) > 0.5
            mask = mask * pred_mask

        if mask.sum() == 0:
            continue

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        pred_depths_valid.append(pred_depth)
        gt_depths_valid.append(gt_depth)

    ratio = 1.0
    if scale_depth:
        ratio = np.median(np.concatenate(gt_depths_valid)) / \
                    np.median(np.concatenate(pred_depths_valid))
    
    for i in range(len(pred_depths_valid)):
        gt_depth = gt_depths_valid[i]
        pred_depth = pred_depths_valid[i]

        pred_depth *= ratio
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)

    # print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    # print("\n-> Done!")

    # if savedir is not None:
    #     with open(os.path.join(savedir, 'depth_evaluation.txt'), 'a+') as f:
    #         if len(f.readlines()) == 0:
    #             f.writelines(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + '    scale_depth\n')
    #         f.writelines(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + f"    {scale_depth}   \\\\")
    
    return mean_errors
   
def save_evaluation_results(dir_log_eval, errors_mesh, name_exps, step_evaluation):
    # save evaluation results to latex format
    mean_errors_mesh = errors_mesh.mean(axis=0)     # 4*7

    names_log = ['err_gt_mesh', 'err_gt_mesh_scale', 'err_gt_depth', 'err_gt_depth_scale']
    dir_log_eval = f'{dir_log_eval}/{step_evaluation:08d}'
    IOUtils.ensure_dir_existence(dir_log_eval)
    for idx_errror_type in range(4):
        with open(f'{dir_log_eval}/{names_log[idx_errror_type]}.txt', 'w') as f_log:
            len_name = len(name_exps[0][0])
            f_log.writelines(('No.' + ' '*np.max([0, len_name-len('   scene id')]) + '     scene id     ' + "{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + '\n')
            for idx_scan in range(errors_mesh.shape[0]):
                f_log.writelines((f'[{idx_scan}] {name_exps[idx_scan][0]} ' + ("&{: 8.3f}  " * 7).format(*errors_mesh[idx_scan, idx_errror_type, :].tolist())) + f" \\\ {name_exps[idx_scan][1]}\n")
            
            f_log.writelines((' '*len_name + 'Mean' + " &{: 8.3f} " * 7).format(*mean_errors_mesh[idx_errror_type, :].tolist()) + " \\\ \n")

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

    return indices, distances

def evaluate_geometry_neucon(file_pred, 
                             file_trgt, 
                             threshold=.05, 
                             down_sample=.02):
    """ Borrowed from NeuralRecon
    Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """

    pcd_pred = GeoUtils.read_point_cloud(file_pred)
    pcd_trgt = GeoUtils.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    ind1, dist1 = nn_correspondance(verts_pred, verts_trgt)  # para2->para1: dist1 is gt->pred
    ind2, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    chamfer= np.mean(dist1**2)+np.mean(dist2**2)
    metrics = {'dist1': np.mean(dist2),  # pred->gt
               'dist2': np.mean(dist1),  # gt -> pred
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               'chamfer': chamfer,
               }

    metrics = np.array([np.mean(dist2), np.mean(dist1), precision, recal, fscore, chamfer])
    logging.info(f'{file_pred.split("/")[-1]}: {metrics}')
    return metrics

def error_mesh(file_trgt,
               file_pred,
               error_bound=0.02):
    mesh_trgt = GeoUtils.read_triangle_mesh(file_trgt)
    verts_trgt = np.asarray(mesh_trgt.vertices)
    triangles_trgt = np.asarray(mesh_trgt.triangles)

    mesh_pred = GeoUtils.read_triangle_mesh(file_pred)
    verts_pred = np.asarray(mesh_pred.vertices)
    triangles_pred = np.asarray(mesh_pred.triangles)

    ind1, dist1 = nn_correspondance(verts_pred, verts_trgt)  # para2->para1: dist1 is gt->pred
    ind2, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    dist2_copy = dist2.copy()
    dist2_copy[ind1]=dist1
    dist=dist2**2+dist2_copy**2

    dist_score = dist.clip(0, error_bound) / error_bound
    color_map = cm.get_cmap('Reds')
    colors = color_map(dist_score)[:, :3]

    path_mesh_pred_error = IOUtils.add_file_name_suffix(file_pred, f'_error_{error_bound}')
    logging.info(f'print error mesh: {path_mesh_pred_error}')
    GeoUtils.save_mesh(path_mesh_pred_error, verts_pred, triangles_pred, colors) 

def evaluate_3D_mesh_neuris(path_mesh_pred, 
                     scene_name, 
                     dir_dataset = './Data/dataset/indoor',
                     eval_threshold = 0.05, 
                     reso_level = 2.0, 
                     check_existence = True):
    '''
    1. clean extra part
    2. Evaluate geometry quality of neus using Precison, Recall and F-score.
    '''
    dir_scan = f'{dir_dataset}/{scene_name}'
    path_intrin = f'{dir_scan}/intrinsic_depth.txt'
    target_img_size = (640, 480)
    dir_poses = f'{dir_scan}/pose'
    
    path_mesh_gt = f'{dir_scan}/{scene_name}_vh_clean_2.ply'
    path_mesh_gt_clean = IOUtils.add_file_name_suffix(path_mesh_gt, '_clean')
    path_mesh_gt_2dmask = f'{dir_scan}/{scene_name}_vh_clean_2_2dmask.npz'
    
    # (1) clean GT mesh
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_gt_clean, path_mesh_gt, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                check_existence = check_existence)
    GeoUtils.generate_mesh_2dmask(path_mesh_gt_2dmask, path_mesh_gt_clean, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                check_existence = check_existence)
    # for fair comparison
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_gt_clean, path_mesh_gt, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                path_mask_npz=path_mesh_gt_2dmask,
                                                check_existence = check_existence)

    # (2) clean predicted mesh
    path_mesh_pred_clean_bbox = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox')
    path_mesh_pred_clean_bbox_faces = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox_faces')
    path_mesh_pred_clean_bbox_faces_mask = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox_faces_mask')

    # 在validate_mesh中已经做了第一步
    GeoUtils.clean_mesh_points_outside_bbox(path_mesh_pred_clean_bbox, 
                                            path_mesh_pred, 
                                            path_mesh_gt,
                                            scale_bbox=1.1,
                                            check_existence = check_existence)
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_pred_clean_bbox_faces, 
                                              path_mesh_pred_clean_bbox, 
                                              path_intrin, 
                                              dir_poses, 
                                              target_img_size, 
                                              reso_level=reso_level,
                                              check_existence = check_existence)
    GeoUtils.clean_mesh_points_outside_frustum(path_mesh_pred_clean_bbox_faces_mask, 
                                               path_mesh_pred_clean_bbox_faces, 
                                               path_intrin, 
                                               dir_poses, 
                                               target_img_size, 
                                               reso_level=reso_level,
                                               path_mask_npz=path_mesh_gt_2dmask,
                                               check_existence = check_existence)
    
    path_eval = path_mesh_pred_clean_bbox_faces_mask 
    metrices_eval = evaluate_geometry_neucon(path_eval, 
                                             path_mesh_gt_clean,             
                                             threshold=eval_threshold, 
                                             down_sample=.02) #f'{dir_eval_fig}/{scene_name}_step{iter_step:06d}_thres{eval_threshold}.png')

    return metrices_eval

def evaluate_3D_mesh_TSDF(path_mesh_pred, 
                            scene_name, 
                            dir_dataset = './Data/dataset/indoor',
                            eval_threshold = [0.05], 
                            check_existence = True):
    '''
    1. construct a TSDF mesh
    2. evaluate geometry quality of neus using Precison, Recall and F-score.
    '''
    dir_scan = f'{dir_dataset}/{scene_name}'
    target_img_size = (640, 480)

    # 1.construct TSDF of GT mesh
    # todo: scannetpp
    path_mesh_gt = f'{dir_dataset}/{scene_name}/{scene_name}_vh_clean_2.ply'
    # path_mesh_gt = f'{dir_dataset}/{scene_name}/mesh.ply'
    
    path_mesh_gt_TSDF = IOUtils.add_file_name_suffix(path_mesh_gt, '_TSDF')
    logging.info(f'Constructing TSDF of GT mesh: {path_mesh_gt}')
    TSDFUtils.construct_TSDF(path_mesh_gt,
                        path_mesh_gt_TSDF,
                        scene_name=scene_name,
                        dir_scan=dir_scan,
                        target_img_size=target_img_size,
                        check_existence=check_existence)
    
    # 2.construct TSDF of predicted mesh
    path_mesh_pred_TSDF = IOUtils.add_file_name_suffix(path_mesh_pred, '_TSDF')
    logging.info(f'Constructing TSDF of predicted mesh: {path_mesh_pred}')
    TSDFUtils.construct_TSDF(path_mesh_pred,
                    path_mesh_pred_TSDF,
                    scene_name=scene_name,
                    dir_scan=dir_scan,
                    target_img_size=target_img_size,
                    check_existence=check_existence)
    
    error_mesh(path_mesh_gt, path_mesh_pred, error_bound=0.009)
    metrices_eval=[]
    # for thredhold_i in eval_threshold:
    #     metrices = evaluate_geometry_neucon(path_mesh_pred_TSDF, 
    #                                         path_mesh_gt_TSDF, 
    #                                         threshold=thredhold_i, 
    #                                         down_sample=.02)
    #     metrices_eval.append(metrices[-2])
    # metrices_eval.append(metrices[-1])
    for thredhold_i in eval_threshold:
        metrices = evaluate_geometry_neucon(path_mesh_pred_TSDF, 
                                            path_mesh_gt_TSDF, 
                                            threshold=thredhold_i, 
                                            down_sample=.02)
    metrices_eval = metrices

    return metrices_eval

def compute_chamfer(pcd_pred, pcd_gt, colour_map_np, draw_label=False, Manhattan=False):
    verts_pred = np.asarray(pcd_pred.points)
    verts_gt = np.asarray(pcd_gt.points)
    colors_gt =np.asarray(pcd_gt.colors)
    labels_gt=np.zeros(colors_gt.shape[0])
    
    #标签索引
    for idx in tqdm(range(colour_map_np.shape[0]), desc='indexing label...'):
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
    
    dist_t = np.array(dist_t)**2
    dist_p = np.array(dist_p)**2
    
    chamfer=np.mean(dist_t)+np.mean(dist_p)
    # gt->pred
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
    
    # pred->gt
    indices_pred_wall= (labels_gt[indices_p]==1) 
    indices_pred_floor= (labels_gt[indices_p]==2)
    indices_pred_other= (~indices_pred_wall & ~indices_pred_floor)
    assert indices_pred_wall.sum()+indices_pred_floor.sum()+indices_pred_other.sum()==dist_p.shape[0]
    
    chamfel_wall+=np.mean(dist_p[indices_pred_wall])
    chamfel_floor+=np.mean(dist_p[indices_pred_floor])
    chamfel_other+=np.mean(dist_p[indices_pred_other])

    return np.array([chamfer, chamfel_wall, chamfel_floor, chamfel_other])*100

def eval_chamfer(path_mesh_pred, 
                     scene_name, 
                     dir_dataset = './Data/dataset/indoor',
                     down_sample=0.02,
                     MANHATTAN=False):
    dir_scan = f'{dir_dataset}/{scene_name}'
    target_img_size = (640, 480)

    # read points
    # todo: scannetpp
    path_mesh_gt = f'{dir_dataset}/{scene_name}/{scene_name}_vh_clean_2.labels.ply'
    # path_mesh_gt = f'{dir_dataset}/{scene_name}/mesh.ply'
    
    path_mesh_pred_TSDF = IOUtils.add_file_name_suffix(path_mesh_pred, '_TSDF')
    logging.info(f'Constructing TSDF of predicted mesh: {path_mesh_pred}')
    TSDFUtils.construct_TSDF(path_mesh_pred,
                    path_mesh_pred_TSDF,
                    scene_name=scene_name,
                    dir_scan=dir_scan,
                    target_img_size=target_img_size,
                    check_existence=True)

    pcd_gt = GeoUtils.read_point_cloud(path_mesh_gt)
    pcd_pred = GeoUtils.read_point_cloud(path_mesh_pred_TSDF)

    if down_sample:
        pcd_gt = pcd_gt.voxel_down_sample(down_sample)
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)

    # compute chamfer distance
    colour_map_np = NYUUtils.nyu40_colour_code
    chamfer_metric = compute_chamfer(pcd_pred, pcd_gt, colour_map_np, draw_label=False, Manhattan=MANHATTAN)

    return chamfer_metric
     
def save_evaluation_results_to_latex(path_log, 
                                        header = '                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                        results = None, 
                                        names_item = None, 
                                        save_mean = None, 
                                        mode = 'w',
                                        precision = 3):
    '''Save evaluation results to txt in latex mode
    Args:
        header:
            for F-score: '                     Accu.      Comp.      Prec.     Recall     F-score \n'
        results:
            narray, N*M, N lines with M metrics
        names_item:
            N*1, item name for each line
        save_mean: 
            whether calculate the mean value for each metric
        mode:
            write mode, default 'w'
    '''
    # save evaluation results to latex format
    with open(path_log, mode) as f_log:
        if header:
            f_log.writelines(header)
        if results is not None:
            num_lines, num_metrices = results.shape
            if names_item is None:
                names_item = np.arange(results.shape[0])
            for idx in range(num_lines):
                f_log.writelines((f'{names_item[idx]}    ' + ("&{: 8.3f}  " * num_metrices).format(*results[idx, :].tolist())) + " \\\ \n")
        if save_mean:
            mean_results = results.mean(axis=0)     # 4*7
            mean_results = np.round(mean_results, decimals=precision)
            f_log.writelines(( '       Mean    ' + " &{: 8.3f} " * num_metrices).format(*mean_results[:].tolist()) + " \\\ \n")
 
def save_evaluation_results_to_markdown(path_log, 
                                        header = '                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                        name_baseline=None,
                                        results = None, 
                                        names_item = None, 
                                        save_mean = True, 
                                        mode = 'w',
                                        precision = 3):
    '''Save evaluation results to txt in latex mode
    Args:
        header:
            for F-score: '                     Accu.      Comp.      Prec.     Recall     F-score \n'
        results:
            narray, N*M, N lines with M metrics
        names_item:
            N*1, item name for each line
        save_mean: 
            whether calculate the mean value for each metric
        mode:
            write mode, default 'w'
    '''
    # save evaluation results to latex format
    results=np.array(results)
    with open(path_log, mode) as f_log:
        if header:
            f_log.writelines(header)
        if results is not None:
            num_lines, num_metrices = results.shape
            if names_item is None:
                names_item = np.arange(results.shape[0])
            for idx in range(num_lines):
                f_log.writelines((f'|{names_item[idx]}  | {name_baseline}|' + ("{: 8.3f}|" * num_metrices).format(*results[idx, :].tolist())) + " \n")
        if save_mean:
            mean_results = np.nanmean(results,axis=0)     # 4*7
            mean_results = np.round(mean_results, decimals=precision)
            f_log.writelines(( f'|       Mean  | {name_baseline}|' + "{: 8.3f}|" * num_metrices).format(*mean_results[:].tolist()) + " \n")

def save_evaluation_results_to_txt(path_log, 
                                        header = '                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                        exp_name=None,
                                        results = None, 
                                        names_item = None, 
                                        save_mean = None, 
                                        mode = 'w',
                                        precision = 3):
    '''Save evaluation results to txt in latex mode
    Args:
        header:
            for F-score: '                     Accu.      Comp.      Prec.     Recall     F-score \n'
        results:
            narray, N*M, N lines with M metrics
        names_item:
            N*1, item name for each line
        save_mean: 
            whether calculate the mean value for each metric
        mode:
            write mode, default 'w'
    '''
    # save evaluation results to latex format
    results=np.array(results)
    with open(path_log, mode) as f_log:
        if header:
            f_log.writelines(header)
        if results is not None:
            num_lines, num_metrices = results.shape
            if names_item is None:
                names_item = np.arange(results.shape[0])
            for idx in range(num_lines):
                f_log.writelines((f'{names_item[idx]} {exp_name}\n' + ("{: 6.3f}" * num_metrices).format(*results[idx, :].tolist())) + " \n")
        if save_mean:
            mean_results = np.nanmax(results,axis=0)     # 4*7
            mean_results = np.round(mean_results, decimals=precision)
            f_log.writelines(( f'    Mean {exp_name}\n' + "{: 6.3f}" * num_metrices).format(*mean_results[:].tolist()) + " \n")