import os, argparse, logging
from datetime import datetime
import numpy as np
import cv2
from matplotlib import cm
import open3d as o3d

import preprocess.neuris_data  as neuris_data
import evaluation.EvalScanNet as EvalScanNet
from evaluation.renderer import render_depthmaps_pyrender

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils
import utils.utils_semantic as SemanticUtils

from confs.path import lis_name_scenes

cv2.destroyAllWindows
if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    
    #3D_error_mesh
    #eval_3D_mesh_metrics
    #eval_semantic_2D_metrices
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='3D_error_mesh')
    parser.add_argument('--semantic_class', type=int, default=3,help='number of semantic class')

    args = parser.parse_args()

    if args.mode == 'eval_3D_mesh_metrics':
        dir_dataset = '../Data/dataset/indoor'
        exp_name='semantic_3_test3'
        name_baseline = f'{exp_name}'
        
        eval_threshold = 0.05
        check_existence = True
        
        dir_results_baseline = f'../exps/evaluation/'

        metrics_eval_all = []
        for scene_name in lis_name_scenes:
            logging.info(f'\n\nProcess: {scene_name}')
            # ./exps/evaluation/neus/scene_name
            path_mesh_pred = f'{dir_results_baseline}/{name_baseline}/{scene_name}.ply'#ply
            metrics_eval =  EvalScanNet.evaluate_3D_mesh(path_mesh_pred, scene_name, dir_dataset = dir_dataset,
                                                                eval_threshold = 0.05, reso_level = 2, 
                                                                check_existence = check_existence)
    
            metrics_eval_all.append(metrics_eval)
        metrics_eval_all = np.array(metrics_eval_all)
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log = f'{dir_results_baseline}/{name_baseline}/eval_{name_baseline}_3Dmesh_thres{eval_threshold}_{str_date}_markdown.txt'
        
        latex_header=f'{exp_name}\n  scene_name         Accu.      Comp.      Prec.     Recall     F-score \n'
        markdown_header=f'\n| scene_name   |    Method|    Accu.|    Comp.|    Prec.|   Recall|  F-score| \n'
        markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- | ------- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        exp_name=exp_name,
                                                        results = metrics_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'w')

    if args.mode == 'eval_semantic_2D_metrices':
        data_dir='../Data/dataset/indoor'
        exp_dir='../exps/indoor/neus'
        metrics_eval_all = []
        numclass=3
        GT_name='semantic_3'
        name_baseline='semantic_3_test13'
        exp_name = name_baseline
        flag_old=False
        
        dir_results_baseline = f'../exps/evaluation'

        for scene_name in lis_name_scenes:
            logging.info(f'eval semantic: {scene_name}')
            #dir
            GT_dir=os.path.join(data_dir,scene_name,GT_name)
  
            if flag_old:
                render_dir=os.path.join(exp_dir,scene_name,name_baseline,'semantic_render')
            else:
                render_dir=os.path.join(exp_dir,scene_name,name_baseline,'semantic_npz')

            # render_dir=os.path.join(data_dir,scene_name,'semantic_deeplab')
            GT_list=os.listdir(GT_dir)
            id_list=[int(os.path.splitext(frame)[0]) for frame in GT_list]
            id_list=sorted(id_list)
            semantic_GT_list=[]
            semantic_render_list=[]
            
            logging.info(f'loading {scene_name} semantic')
            for idx in id_list:
                GT_file=os.path.join(GT_dir, '%d.png'%idx)
                
                if flag_old:
                    render_file=os.path.join(render_dir, '00160000_'+'0'*(4-len(str(idx)))+str(idx)+'_reso2.png')
                    # render_file=os.path.join(render_dir, str(idx)+'.png')#deeplab
                else:
                    render_file=os.path.join(render_dir, '00160000_'+'0'*(4-len(str(idx)))+str(idx)+'_reso2.npz')

                semantic_GT=cv2.imread(GT_file)[:,:,0]
                
                if flag_old:
                    semantic_render=cv2.imread(render_file)[:,:,0]/80
                else:
                    semantic_render=(np.load(render_file)['arr_0'])

                reso=semantic_GT.shape[0]/semantic_render.shape[0]
                if reso>1:
                    semantic_GT=cv2.resize(semantic_GT, (semantic_render.shape[1],semantic_render.shape[0]), interpolation=cv2.INTER_NEAREST)
                semantic_GT_list.append(semantic_GT)
                semantic_render_list.append(semantic_render)
            
            acc_mean,acc_cls,acc_cls_mean, iou,iou_mean,iou_freq_mean=SemanticUtils.eval_semantic(semantic_GT_list,semantic_render_list,numclass)
            
            acc_list=np.append(acc_cls,[acc_cls_mean])
            iou_list=np.append(iou,[iou_mean])
            metrics_eval=np.append(acc_list,iou_list)
            # metrics_eval=[acc_cls_mean,iou_mean]
            metrics_eval_all.append(metrics_eval)

            logging.info(f'{scene_name}: {metrics_eval}')

        metrics_eval_all = np.array(metrics_eval_all)
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log = f'{dir_results_baseline}/{name_baseline}/eval_{name_baseline}_semantic_{str_date}_markdown.txt'
        
        markdown_header=f'3D mesh evaluation\n| scene_ name   |   Method|  Acc_o|  Acc_w|  Acc_f| Acc_mean| IoU_o| IoU_w| IoU_f| IoU_mean| \n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        exp_name=exp_name,
                                                        results = metrics_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'w')
            
    if args.mode == 'eval_mesh_2D_metrices':
        dir_dataset = '../Data/dataset/indoor'
        
        exp_name='neuris'
        name_baseline = f'{exp_name}_refuse'
        eval_type_baseline = 'mesh'
        scale_depth = False

        dir_results_baseline = f'../exps/evaluation/{name_baseline}'
        results_all =  []
        for scene_name in lis_name_scenes:
            # scene_name += '_corner'
            print(f'Processing {scene_name}...')
            path_intrin = f'{dir_dataset}/{scene_name}/intrinsic_depth.txt'
            dir_scan = f'{dir_dataset}/{scene_name}'
            if eval_type_baseline == 'mesh':
                # use rendered depth map
                path_mesh_baseline =  f'{dir_results_baseline}/{scene_name}.ply'
                pred_depths = render_depthmaps_pyrender(path_mesh_baseline, path_intrin, 
                                                            dir_poses=f'{dir_scan}/pose')
                img_names = IOUtils.get_files_stem(f'{dir_scan}/depth', '.png')
            elif eval_type_baseline == 'depth':
                dir_depth_baseline =  f'{dir_results_baseline}/{scene_name}'
                pred_depths = GeoUtils.read_depth_maps_np(dir_depth_baseline)
                img_names = IOUtils.get_files_stem(dir_depth_baseline, '.npy')
                
            # evaluation
            dir_gt_depth = f'{dir_scan}/depth'
            gt_depths, _ = EvalScanNet.load_gt_depths(img_names, dir_gt_depth)
            err_gt_depth_scale = EvalScanNet.depth_evaluation(gt_depths, pred_depths, dir_results_baseline, scale_depth=scale_depth)
            results_all.append(err_gt_depth_scale)
            
        results_all = np.array(results_all)
        print('All results: ', results_all)

        count = 0
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log_all = f'{dir_results_baseline}/eval_{name_baseline}_depth-scale_{scale_depth}_{eval_type_baseline}_{str_date}.txt'
        # EvalScanNet.save_evaluation_results_to_markdown(path_log_all, header = f'{str_date}\n\n',  mode = 'w')
        
        precision = 3
        results_all = np.round(results_all, decimals=precision)
        markdown_header=f'depth evaluation\n| scene_ name   |   Method|  abs_rel|  sq_rel|  rmse| rmse_log| a1| a2| a3| \n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- | ----- | ----- | ----- |\n' 
        EvalScanNet.save_evaluation_results_to_markdown(path_log_all, 
                                                        header = markdown_header, 
                                                        exp_name=exp_name,
                                                        results = results_all, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'w',
                                                        precision = precision)
    if args.mode=='3D_error_mesh':
        logging.info('output 3D error mesh.')
        dir_dataset = '../Data/dataset/indoor'
        dir_results_baseline = f'../exps/evaluation/'
        exp_name='semantic_3_test17'
        name_baseline = f'{exp_name}_refuse'
        
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


    if args.mode == 'evaluate_normal':
        # compute normal errors
        exp_name = 'exp_neuris'
        name_normal_folder = 'normal_render'
        
        dir_root_dataset = './dataset/indoor'
        dir_root_normal_gt = '../TiltedImageSurfaceNormal/datasets/scannet-frames'

        err_neus_all, err_pred_all = [], []
        num_imgs_eval_all = 0
        
        for scene_name in lis_name_scenes:
            # scene_name = 'scene0085_00'
            print(f'Process: {scene_name}')
           
            dir_normal_neus = f'./exps/indoor/neus/{scene_name}/{exp_name}/{name_normal_folder}'
            
            dir_normal_pred = f'{dir_root_dataset}/{scene_name}/pred_normal' 
            dir_poses = f'{dir_root_dataset}/{scene_name}/pose' 
            dir_normal_gt = f'{dir_root_normal_gt}/{scene_name}'
            error_neus, error_pred, num_imgs_eval = NormalUtils.evauate_normal(dir_normal_neus, dir_normal_pred, dir_normal_gt, dir_poses)
            err_neus_all.append(error_neus)
            err_pred_all.append(error_pred)
            
            num_imgs_eval_all += num_imgs_eval

        error_neus_all = np.concatenate(err_neus_all).reshape(-1)
        err_pred_all = np.concatenate(err_pred_all).reshape(-1)
        metrics_neus = NormalUtils.compute_normal_errors_metrics(error_neus_all)
        metrics_pred = NormalUtils.compute_normal_errors_metrics(err_pred_all)
        NormalUtils.log_normal_errors(metrics_neus, first_line='metrics_neus fianl')
        NormalUtils.log_normal_errors(metrics_pred, first_line='metrics_pred final')
        print(f'Total evaluation images: {num_imgs_eval_all}')

    if args.mode == 'evaluate_nvs':
           # compute normal errors
        name_img_folder = 'image_render'
        sample_interval = 1

        exp_name_nerf = 'exp_nerf'
        exp_name_neuris = 'exp_neuris'
        exp_name_neus  = 'exp_neus'

        evals_nvs_all ={
            'nerf': exp_name_nerf,
            'neus': exp_name_neus,
            'neuris': exp_name_neuris
        }
        psnr_all_methods = {}
        psnr_imgs = []
        psnr_imgs_stem = []
        np.set_printoptions(precision=3)
        for key in evals_nvs_all:
            exp_name = evals_nvs_all[key]
            model_type = 'nerf' if key == 'nerf' else 'neus'

            print(f"Start to eval: {key}. {exp_name}")
            err_neus_all, err_pred_all = [], []
            num_imgs_eval_all = 0
            
            psnr_scenes_all = []
            psnr_mean_all = []
            for scene_name in lis_name_scenes:
                # scene_name = 'scene0085_00'
                scene_name = scene_name + '_nvs'
                print(f'Process: {scene_name}')
                
                dir_img_gt = f'./dataset/indoor/{scene_name}/image'
                dir_img_neus = f'./exps/indoor/{model_type}/{scene_name}/{exp_name}/{name_img_folder}'
                psnr_scene, vec_stem_eval = ImageUtils.eval_imgs_psnr(dir_img_neus, dir_img_gt, sample_interval)
                print(f'PSNR: {scene_name} {psnr_scene.mean()}  {psnr_scene.shape}')
                psnr_scenes_all.append(psnr_scene)
                psnr_imgs.append(psnr_scene)
                psnr_imgs_stem.append(vec_stem_eval)
                psnr_mean_all.append(psnr_scene.mean())
                # input("anything to continue")
            
            psnr_scenes_all = np.concatenate(psnr_scenes_all)
            psnr_mean_all = np.array(psnr_mean_all)
            print(f'\n\n Mean of scnes: {psnr_mean_all.mean()}. PSNR of all scenes: {psnr_mean_all} \n mean of images:{psnr_scenes_all.mean()} image numbers: {len(psnr_scenes_all)} ')

            psnr_all_methods[key] = (psnr_scenes_all.mean(), psnr_mean_all.mean(),  psnr_mean_all)
        
        # psnr_imgs = vec_stem_eval + psnr_imgs
        path_log_temp = f'./exps/indoor/evaluation/nvs/evaluation_temp_{lis_name_scenes[0]}.txt'
        flog_temp = open(path_log_temp, 'w')
        for i in range(len(vec_stem_eval)):
            try:
                flog_temp.write(f'{psnr_imgs_stem[0][i][9:13]}  {psnr_imgs_stem[1][i][9:13]}  {psnr_imgs_stem[2][i][9:13]}: {psnr_imgs[0][i]:.1f}:  {psnr_imgs[1][i]:.1f}  {psnr_imgs[2][i]:.1f}\n')
            except Exception:
                print(f'Error: skip {vec_stem_eval[i]}')
                continue
        flog_temp.close()
        input('Continue?')

        print(f'Finish NVS evaluation')
        # print eval information
        path_log = f'./exps/indoor/evaluation/nvs/evaluation.txt'
        flog_nvs = open(path_log, 'a')
        flog_nvs.write(f'sample interval: {sample_interval}. Scenes number: {len(lis_name_scenes)}. Scene names: {lis_name_scenes}\n\n')
        flog_nvs.write(f'Mean_img  mean_scenes  scenes-> \n')
        for key in psnr_all_methods:
            print(f'[{key}] Mean of all images: {psnr_all_methods[key][0]}. Mean of scenes: {psnr_all_methods[key][1]} \n Scenes: {psnr_all_methods[key][2]}\n')
            flog_nvs.write(f'{key:10s} {psnr_all_methods[key][0]:.03f} {psnr_all_methods[key][1]:.03f}. {psnr_all_methods[key][2]}.\n')
        flog_nvs.write(f'Images number: {len(psnr_scenes_all)}\n')
        flog_nvs.close()
    
    print('Done')
