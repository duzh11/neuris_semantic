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

# from confs.path import lis_name_scenes

cv2.destroyAllWindows
if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    
    lis_name_scenes=['scene0378_00']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='evaluate_normal')
    parser.add_argument('--exp_name', type=str, default='deeplab_semantic/ce_stop')
    parser.add_argument('--dir_dataset', type=str, default='../Data/dataset/indoor')
    parser.add_argument('--dir_results_baseline', type=str, default='../exps/indoor/neus')
    parser.add_argument('--acc', type=str, default='fine')
    parser.add_argument('--save_mean', default=False, action="store_true")
    parser.add_argument('--semantic_class', type=int, default=40, help='number of semantic class')
    args = parser.parse_args()

    dir_dataset = args.dir_dataset
    exp_name = args.exp_name
    name_baseline = exp_name.split('/')[-1]
    dir_results_baseline = os.path.join(args.dir_results_baseline, exp_name)

    logging.info(f'Eval mode: {args.mode}')
    if args.mode == 'eval_3D_mesh_neuris':
        eval_threshold = 0.05
        metrics_eval_all = []
        for scene_name in lis_name_scenes:
            logging.info(f'Eval 3D mesh: {scene_name}')
            path_mesh_pred = f'{dir_results_baseline}/{scene_name}/meshes/{scene_name}.ply'
            metrics_eval =  EvalScanNet.evaluate_3D_mesh_neuris(path_mesh_pred, 
                                                         scene_name, 
                                                         dir_dataset = dir_dataset,
                                                         eval_threshold = eval_threshold, 
                                                         reso_level = 2, 
                                                         check_existence = True)
    
            metrics_eval_all.append(metrics_eval)
        metrics_eval_all = np.array(metrics_eval_all)
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log = f'{dir_results_baseline}/{name_baseline}_eval3Dmesh_neuris_thres{eval_threshold}_{str_date}.md'
        
        markdown_header=f'| scene_name   |    Method|    Accu.|    Comp.|    Prec.|   Recall|  F-score|  Chamfer\n'
        markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- | ------- | ------- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metrics_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'w')  

    if args.mode == 'eval_3D_mesh_TSDF':
        eval_threshold_lis = [0.03,0.05,0.07]
        
        metrics_eval_all = []
        for scene_name in lis_name_scenes:
            logging.info(f'Eval 3D mesh: {scene_name}')
            path_mesh_pred = f'{dir_results_baseline}/{scene_name}/meshes/{scene_name}.ply'
            metrics_eval =  EvalScanNet.evaluate_3D_mesh_TSDF(path_mesh_pred,
                                                              scene_name,
                                                              dir_dataset = '../Data/dataset/indoor',
                                                              eval_threshold = eval_threshold_lis,
                                                              reso_level = 2.0, 
                                                              check_existence = True)
            
            metrics_eval_all.append(metrics_eval)
        metrics_eval_all = np.array(metrics_eval_all)

        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log = f'{dir_results_baseline}/{name_baseline}_eval3Dmesh_TSDF_{str_date}.md'
        
        markdown_header='Eval mesh\n| scene_name   |    Method| F-score$_{0.03}$| F-score$_{0.05}$| F-score$_{0.07}$| Chamfer|\n'
        markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metrics_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'w')

    if args.mode == 'eval_mesh_2D_metrices':
        eval_type_baseline = 'depth'
        scale_depth = False
        results_all =  []

        for scene_name in lis_name_scenes:
            logging.info(f'Processing {scene_name}...')
            
            dir_scan = f'{dir_dataset}/{scene_name}'
            path_intrin = f'{dir_scan}/intrinsic_depth.txt'
            if eval_type_baseline == 'mesh':
                # use rendered depth map
                path_mesh_baseline =  f'{dir_results_baseline}/{scene_name}/meshes/{scene_name}.ply'
                pred_depths = render_depthmaps_pyrender(path_mesh_baseline, path_intrin, 
                                                            dir_poses=f'{dir_scan}/pose')
            elif eval_type_baseline == 'depth':
                dir_depth_baseline =  f'{dir_results_baseline}/{scene_name}/depth/{args.acc}'
                pred_depths = GeoUtils.read_depth_maps_np(dir_depth_baseline)
            
            # evaluation
            img_names = IOUtils.get_files_stem(f'{dir_scan}/depth', '.png')
            dir_gt_depth = f'{dir_scan}/depth'
            gt_depths, _ = EvalScanNet.load_gt_depths(img_names, dir_gt_depth, pred_depths.shape[1], pred_depths.shape[2])
            err_gt_depth_scale = EvalScanNet.depth_evaluation(gt_depths, pred_depths, dir_results_baseline, scale_depth=scale_depth)
            results_all.append(err_gt_depth_scale)
            
        results_all = np.array(results_all)

        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log = f'{dir_results_baseline}/{name_baseline}_evaldepth_{args.acc}_{scale_depth}_{eval_type_baseline}_{str_date}.md'
        
        precision = 3
        results_all = np.round(results_all, decimals=precision)
        markdown_header=f'depth evaluation\n| scene_ name   |   Method|  abs_rel|  sq_rel|  rmse| rmse_log| a1| a2| a3| \n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- | ----- | ----- | ----- |\n' 
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = results_all, 
                                                        names_item = lis_name_scenes, 
                                                        save_mean = True, 
                                                        mode = 'w')
    
    # if args.mode == 'evaluate_semantic':


    #######-------to be modified----------
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
