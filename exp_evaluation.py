import os, logging,argparse
from datetime import datetime
import numpy as np
import cv2
from matplotlib import cm
import open3d as o3d

import evaluation.EvalScanNet as EvalScanNet
from evaluation.renderer import render_depthmaps_pyrender

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils
import utils.utils_semantic as SemanticUtils

from confs.path import lis_name_scenes

MANHATTAN=False

cv2.destroyAllWindows
str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    
    parser = argparse.ArgumentParser()
    # eval_3D_mesh_neuris， eval_3D_mesh_TSDF， eval_chamfer， eval_mesh_2D_metrices， eval_semantic2D, 
    parser.add_argument('--mode', type=str, default='evaluate_nvs')
    parser.add_argument('--exp_name', type=str, default='deeplab_ce/ce_final')
    parser.add_argument('--dir_dataset', type=str, default='../Data/dataset/indoor')
    parser.add_argument('--dir_results_baseline', type=str, default='../exps/indoor/neus')
    parser.add_argument('--acc', type=str, default='fine')
    parser.add_argument('--iter', type=int, default=160000, help='iter')
    parser.add_argument('--semantic_class', type=int, default=40, help='number of semantic class')
    args = parser.parse_args()

    dir_dataset = args.dir_dataset
    exp_name = args.exp_name
    name_baseline = exp_name.split('/')[-1]
    dir_results_baseline = os.path.join(args.dir_results_baseline, exp_name)

    logging.info(f'-----Eval mode: {args.mode}-----')
    data_mode_list = ['train']
    if args.mode == 'eval_3D_mesh_neuris':
        eval_threshold = 0.05
        metrics_eval_all = []
        for scene_name in lis_name_scenes:
            logging.info(f'Processing: {scene_name}')
            path_mesh_pred = f'{dir_results_baseline}/{scene_name}/meshes/{scene_name}_clean_bbox.ply'
            metrics_eval =  EvalScanNet.evaluate_3D_mesh_neuris(path_mesh_pred, 
                                                         scene_name, 
                                                         dir_dataset = dir_dataset,
                                                         eval_threshold = eval_threshold, 
                                                         reso_level = 2, 
                                                         check_existence = True)
    
            metrics_eval_all.append(metrics_eval)
        metrics_eval_all = np.array(metrics_eval_all)
        path_log = f'{dir_results_baseline}/{name_baseline}_eval3Dmesh_neuris_thres{eval_threshold}_{str_date}.md'
        
        markdown_header=f'| scene_name   |    Method|    Accu.|    Comp.|    Prec.|   Recall|  F-score|  Chamfer\n'
        markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- | ------- | ------- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metrics_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        mode = 'w')  

    if args.mode == 'eval_3D_mesh_TSDF':
        eval_threshold_lis = [0.05]
        
        metrics_eval_all = []
        for scene_name in lis_name_scenes:
            logging.info(f'Processing: {scene_name}')
            path_mesh_pred = f'{dir_results_baseline}/{scene_name}/meshes/{scene_name}_clean_bbox.ply'
            metrics_eval =  EvalScanNet.evaluate_3D_mesh_TSDF(path_mesh_pred,
                                                              scene_name,
                                                              dir_dataset = '../Data/dataset/indoor',
                                                              eval_threshold = eval_threshold_lis)
            
            metrics_eval_all.append(metrics_eval)
        metrics_eval_all = np.array(metrics_eval_all)

        path_log = f'{dir_results_baseline}/{name_baseline}_eval3Dmesh_TSDF_{str_date}.md'
        
        markdown_header=f'| scene_name   |    Method|    Accu.|    Comp.|    Prec.|   Recall|  F-score|  Chamfer\n'
        markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- | ------- | ------- |\n'
        # markdown_header='Eval mesh\n| scene_name   |    Method| F-score$_{0.03}$| F-score$_{0.05}$| F-score$_{0.07}$| Chamfer|\n'
        # markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metrics_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        mode = 'w')    
    
    if args.mode == 'eval_chamfer':
        metrics_eval_all = []

        for scene_name in lis_name_scenes:
            logging.info(f'Processing: {scene_name}')
            path_mesh_pred = f'{dir_results_baseline}/{scene_name}/meshes/{scene_name}_clean_bbox.ply'

            metric_eval = EvalScanNet.eval_chamfer(path_mesh_pred, 
                                                   scene_name, 
                                                   dir_dataset = '../Data/dataset/indoor',
                                                   MANHATTAN=MANHATTAN)
            metrics_eval_all.append(metric_eval)

        path_log = f'{dir_results_baseline}/{name_baseline}_evalchamfer_{str_date}.md'

        markdown_header='Eval mesh\n| scene_name   |    Method| CD| Wall| Floor| Other|\n'
        markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metrics_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        mode = 'w')

    if args.mode == 'eval_mesh_2D_metrices':
        eval_type_baseline = 'depth'
        scale_depth = False
        metric_train_all =  []
        metric_test_all = []

        for scene_name in lis_name_scenes:
            logging.info(f'Processing {scene_name}...')
            
            dir_scan = f'{dir_dataset}/{scene_name}'
            path_intrin = f'{dir_scan}/intrinsic_depth.txt'

            for data_mode in data_mode_list:
                if eval_type_baseline == 'mesh':
                    # use rendered depth map
                    path_mesh_baseline =  f'{dir_results_baseline}/{scene_name}/meshes/{scene_name}.ply'
                    pred_depths = render_depthmaps_pyrender(path_mesh_baseline, path_intrin, 
                                                                dir_poses=f'{dir_scan}/pose/{data_mode}')
                elif eval_type_baseline == 'depth':
                    dir_depth_baseline =  f'{dir_results_baseline}/{scene_name}/depth/{data_mode}/{args.acc}'
                    pred_depths = GeoUtils.read_depth_maps_np(dir_depth_baseline, args.iter)
                
                # evaluation
                img_names = IOUtils.get_files_stem(f'{dir_scan}/depth/{data_mode}', '.png')
                dir_gt_depth = f'{dir_scan}/depth/{data_mode}'
                gt_depths, _ = EvalScanNet.load_gt_depths(img_names, dir_gt_depth, pred_depths.shape[1], pred_depths.shape[2])
                err_gt_depth_scale = EvalScanNet.depth_evaluation(gt_depths, pred_depths, dir_results_baseline, scale_depth=scale_depth)
                
                if data_mode == 'train':
                    metric_train_all.append(err_gt_depth_scale)
                elif data_mode == 'test':
                    metric_test_all.append(err_gt_depth_scale)

        path_log = f'{dir_results_baseline}/{name_baseline}_evaldepth_{args.acc}_{scale_depth}_{eval_type_baseline}_{args.iter}_{str_date}.md'
        
        precision = 3
        metric_eval_all = np.round(np.array(metric_train_all), decimals=precision)
        markdown_header=f'train\n| scene_ name   |   Method|  abs_rel|  sq_rel|  rmse| rmse_log| a1| a2| a3| \n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- | ----- | ----- | ----- |\n' 
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metric_eval_all, 
                                                        names_item = lis_name_scenes, 
                                                        mode = 'w')
        
        # metric_eval_all = np.round(np.array(metric_test_all), decimals=precision)
        # markdown_header=f'\ntest\n| scene_ name   |   Method|  abs_rel|  sq_rel|  rmse| rmse_log| a1| a2| a3| \n'
        # markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- | ----- | ----- | ----- |\n' 
        # EvalScanNet.save_evaluation_results_to_markdown(path_log, 
        #                                                 header = markdown_header, 
        #                                                 name_baseline=name_baseline,
        #                                                 results = metric_eval_all, 
        #                                                 names_item = lis_name_scenes, 
        #                                                 mode = 'a')
    
    if args.mode == 'eval_semantic_2D':
        metric_train_all = []
        metric_test_all = []
        semantic_class = args.semantic_class

        for scene_name in lis_name_scenes:
            logging.info(f'Processing {scene_name}...')
            
            for data_mode in data_mode_list:
                dir_scan = f'{dir_dataset}/{scene_name}'
                dir_exp = f'{dir_results_baseline}/{scene_name}'

                metric_avg, exsiting_label, class_iou, class_accuray = SemanticUtils.evaluate_semantic_2D(dir_scan, 
                                                                                                        dir_exp,
                                                                                                        args.acc, 
                                                                                                        data_mode,
                                                                                                        semantic_class=semantic_class,
                                                                                                        iter = args.iter,
                                                                                                        MANHATTAN=MANHATTAN)
                if data_mode == 'train':
                    metric_train_all.append(metric_avg)
                elif data_mode == 'test':
                    metric_test_all.append(metric_avg)

                path_log = f'{dir_results_baseline}/{scene_name}/{name_baseline}_evalsemantic2D_{data_mode}_{args.acc}_{args.iter}_{semantic_class}_{MANHATTAN}_{str_date}_markdown.md'
                markdown_header_0=[f' {label} |'for label in exsiting_label]
                markdown_header_1 = '\n| -------------| ---------|'+'---------|'*len(exsiting_label)
                markdown_header=  'IoU\n| scene_ name   |   Method|'+''.join(markdown_header_0)+markdown_header_1+'\n'

                EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                            header = markdown_header, 
                                                            name_baseline=name_baseline,
                                                            results = [class_iou], 
                                                            names_item = [scene_name],  
                                                            save_mean=False,
                                                            mode = 'w')
                
                markdown_header = '\nAcc\n| scene_ name   |   Method|' + ''.join(markdown_header_0)+markdown_header_1+'\n'
                EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                            header = markdown_header, 
                                                            name_baseline=name_baseline,
                                                            results = [class_accuray], 
                                                            names_item = [scene_name],  
                                                            save_mean=False,
                                                            mode = 'a')
        
        path_log = f'{dir_results_baseline}/{name_baseline}_evalsemantic2D_{args.acc}_{args.iter}_{semantic_class}_{MANHATTAN}_{str_date}_markdown.md'
        markdown_header='train\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metric_train_all, 
                                                        names_item = lis_name_scenes,  
                                                        mode = 'w')
        
        # markdown_header='\ntest\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
        # markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
        # EvalScanNet.save_evaluation_results_to_markdown(path_log, 
        #                                                 header = markdown_header, 
        #                                                 name_baseline=name_baseline,
        #                                                 results = metric_test_all, 
        #                                                 names_item = lis_name_scenes,  
        #                                                 mode = 'a')
        
        

    if args.mode == 'eval_semantic_3D':
        metrics_volume_all = []
        metrics_surface_all = []
        semantic_class = args.semantic_class

        for scene_name in lis_name_scenes:
            logging.info(f'Processing {scene_name}...')
            # todo: scannetpp
            file_mesh_trgt = os.path.join(dir_dataset, f'{scene_name}/{scene_name}_vh_clean_2.labels.ply')
            # file_mesh_trgt = os.path.join(dir_dataset, f'{scene_name}/mesh_sem.ply')
            file_pred = os.path.join(dir_results_baseline, f'{scene_name}/meshes')
            
            for semantic_mode in ['volume','surface']:
                file_mesh_pred = os.path.join(file_pred, f'{scene_name}_semantic_{semantic_mode}.ply')
                file_semseg_pred = os.path.join(file_pred, f'{scene_name}_semantic_{semantic_mode}.npz')
                file_mesh_pred_transfer = os.path.join(file_pred, f'{scene_name}_semantic_{semantic_mode}_transfer.ply')
                
                mesh_transfer, metric_avg, exsiting_label, class_iou, class_accuray = SemanticUtils.evaluate_semantic_3D(file_mesh_trgt,
                                                                                                                        file_mesh_pred,
                                                                                                                        file_semseg_pred,
                                                                                                                        semantic_class=semantic_class)
                mesh_transfer.export(file_mesh_pred_transfer)

                if semantic_mode == 'volume':
                    metrics_volume_all.append(metric_avg)
                elif semantic_mode == 'surface':
                    metrics_surface_all.append(metric_avg)

                path_log = f'{dir_results_baseline}/{scene_name}/{name_baseline}_evalsemantic3D_{semantic_mode}_{semantic_class}_{str_date}_markdown.md'
                markdown_header_0=[f' {label} |'for label in exsiting_label]
                markdown_header_1 = '\n| -------------| ---------|'+'---------|'*len(exsiting_label)
                markdown_header=  'IoU\n| scene_ name   |   Method|'+''.join(markdown_header_0)+markdown_header_1+'\n'
                EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                            header = markdown_header, 
                                                            name_baseline=name_baseline,
                                                            results = [class_iou], 
                                                            names_item = [scene_name],  
                                                            save_mean=False,
                                                            mode = 'w')
                
                markdown_header = '\nAcc\n| scene_ name   |   Method|' + ''.join(markdown_header_0)+markdown_header_1+'\n'
                EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                            header = markdown_header, 
                                                            name_baseline=name_baseline,
                                                            results = [class_accuray], 
                                                            names_item = [scene_name],  
                                                            save_mean=False,
                                                            mode = 'a')
        
        path_log = f'{dir_results_baseline}/{name_baseline}_evalsemantic3D_{semantic_class}_{str_date}_markdown.md'
        markdown_header='volume\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metrics_volume_all, 
                                                        names_item = lis_name_scenes,  
                                                        mode = 'w')
        
        markdown_header='\nsurface\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metrics_surface_all, 
                                                        names_item = lis_name_scenes,  
                                                        mode = 'a')
                

    if args.mode == 'evaluate_normal':
        # compute normal errors
        dir_root_normal_gt = '/home/du/Proj/2Dv_DL/FrameNet/src/data'

        err_neus_train_all, err_pred_train_all, num_imgs_eval_train_all = [], [], 0
        err_neus_test_all, err_pred_test_all, num_imgs_eval_test_all = [], [], 0
        metric_neus_scene_all, metric_pred_scene_all = [], []
        
        for scene_name in lis_name_scenes:
            logging.info(f'Evaluating Normal: {scene_name}')
            dir_scan = f'{dir_dataset}/{scene_name}'
            dir_exp = f'{dir_results_baseline}/{scene_name}'

            for data_mode in data_mode_list:
                dir_normal_exp = f'{dir_exp}/normal/{data_mode}/fine'
                dir_normal_pred = f'{dir_scan}/normal/{data_mode}/pred_normal' 
                dir_poses = f'{dir_scan}/pose/{data_mode}' 
                # todo: scannetpp
                dir_normal_gt = f'{dir_root_normal_gt}/{scene_name}/scannet-frames/{scene_name}'
                # dir_normal_gt = f'{dir_scan}/normal/{data_mode}/normal_from_depth_cam' 

                error_neus, error_pred, num_imgs_eval, metrics_neus_lis, metrics_pred_lis \
                                = NormalUtils.evauate_normal(dir_normal_exp, dir_normal_pred, dir_normal_gt, dir_poses)
                
                metric_neus_scene_all.append(metrics_neus_lis)
                metric_pred_scene_all.append(metrics_pred_lis)
                if data_mode == 'train':
                    err_neus_train_all.append(error_neus)
                    err_pred_train_all.append(error_pred)
                    num_imgs_eval_train_all += num_imgs_eval
                elif data_mode == 'test':
                    err_neus_test_all.append(error_neus)
                    err_pred_test_all.append(error_pred)
                    num_imgs_eval_test_all += num_imgs_eval
        
        # compute metrics for all scenes
        error_neus_train_all = np.concatenate(err_neus_train_all).reshape(-1)
        error_pred_train_all = np.concatenate(err_pred_train_all).reshape(-1)
        metrics_neus_train, metrics_neu_train_Lis = NormalUtils.compute_normal_errors_metrics(error_neus_train_all)
        metrics_pred_train, metrics_pred_train_lis = NormalUtils.compute_normal_errors_metrics(error_pred_train_all)

        NormalUtils.log_normal_errors(metrics_neus_train, first_line='metrics_neus fianl')
        NormalUtils.log_normal_errors(metrics_pred_train, first_line='metrics_pred final')
        print(f'Total evaluation images: {num_imgs_eval_train_all}')
        # Save evaluation results to markdown
        metric_neus_scene_all.append(metrics_neu_train_Lis)
        metric_pred_scene_all.append(metrics_pred_train_lis)

        names_item = lis_name_scenes+['all']
        path_log = f'{dir_results_baseline}/{name_baseline}_evalNormal_{args.iter}_{str_date}_markdown.md'
        markdown_header='----train-----\n\nNeus\n| scene_name   |   Method|  mean|  median|  rmse|  a1|  a2|  a3|  a4|  a5|\n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metric_neus_scene_all, 
                                                        names_item = names_item, 
                                                        save_mean = False, 
                                                        mode = 'w')
        
        markdown_header='\n\nPred\n| scene_name   |   Method|  mean|  median|  rmse|  a1|  a2|  a3|  a4|  a5|\n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metric_pred_scene_all, 
                                                        names_item = names_item, 
                                                        save_mean = False, 
                                                        mode = 'a')

    if args.mode == 'evaluate_nvs':

        metric_train_all = []
        metric_test_all = []

        for scene_name in lis_name_scenes:
            logging.info(f'Evaluating NVS: {scene_name}')
            dir_scan = f'{dir_dataset}/{scene_name}'
            dir_exp = f'{dir_results_baseline}/{scene_name}'
            
            for data_mode in data_mode_list:
                dir_img_gt = f'{dir_scan}/image/{data_mode}'
                dir_img_exp = f'{dir_exp}/img/{data_mode}/fine'
                psnr_scene, ssim_scene, lpips_scene, vec_stem_eval = ImageUtils.eval_NVS(dir_img_exp, dir_img_gt)
                logging.info(f'NVS: {scene_name} {psnr_scene.mean()} {ssim_scene.mean()} \
                             {psnr_scene.shape[0]}')

                if data_mode == 'train':
                    metric_train_all.append([psnr_scene.mean(), ssim_scene.mean(), lpips_scene.mean()])
                elif data_mode == 'test':
                    metric_train_all.append([psnr_scene.mean(), ssim_scene.mean(), lpips_scene.mean()])

        path_log = f'{dir_results_baseline}/{name_baseline}_evalNVS_{args.iter}_{str_date}_markdown.md'
        markdown_header='train\n| scene_ name   |   Method|  PSNR|  SSIM|  LPIPS|\n'
        markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- |\n'
        EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=name_baseline,
                                                        results = metric_train_all, 
                                                        names_item = lis_name_scenes,  
                                                        mode = 'w')
        
        
