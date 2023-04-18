import os, argparse, logging
from datetime import datetime
import numpy as np
import os

# import preprocess.neuris_data as neuris_data
import evaluation.EvalScanNet as EvalScanNet
from evaluation.renderer import render_depthmaps_pyrender

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils

# from confs.path import lis_name_scenes


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval_mesh_3D_metrics')
    # parser.add_argument('--mode', type=str, default='eval_errormap')

    args = parser.parse_args()
    
    #
    dir_dataset = f'/home/zhanghaoyu/dataset/indoor'             # dataset
    dir_exp_results = f'/home/zhanghaoyu/project/NeuRIS/exps/indoor/neus'
    path_intrin = f'{dir_dataset}/scannet intrinsics/intrinsic_depth.txt'          # intrinsic
    dir_evaluation_results = f'/home/zhanghaoyu/project/NeuRIS/exps/evaluation'     # evaluation

    # scene_names = ['scene0050_00', 'scene0084_00','scene0114_02', 'scene0603_00','scene0616_00', 'scene0721_00']
    scene_names = ['scene0050_00']
    # scene_names = ['scene0050_00']
    exp_dict = dict.fromkeys(scene_names)
    for scene in scene_names:
        exp_names = os.listdir(f'{dir_exp_results}/{scene}')
        # exp_name = [i for i in exp_names if i[-3] == '_']
        # exp_name.sort(key=lambda x: x[-1])
        exp_name = [i for i in exp_names if len(i)>17]
        exp_dict[scene] = exp_name


    if args.mode == 'eval_mesh_3D_metrics':

        name_baseline = 'neus' # manhattansdf neuris
        eval_threshold = 0.05
        check_existence = True

        metrics_eval_all = []
        exp_name_ls = []

        for scene_name in scene_names:
            for exp_name in exp_dict[scene_name]:
                logging.info(f'\n\nProcess: {scene_name}/{exp_name}')
                exp_name_ls.append(f'{scene_name}/{exp_name}')

                path_mesh_pred = f'{dir_exp_results}/{scene_name}/{exp_name}/meshes/00160000_reso512_{scene_name}_world.ply'
                metrics_eval =  EvalScanNet.evaluate_3D_mesh(path_mesh_pred, scene_name, dir_dataset = dir_dataset,
                                                                eval_threshold = 0.05, reso_level = 2,
                                                                check_existence = check_existence)
                metrics_eval_all.append(metrics_eval)


        metrics_eval_all = np.array(metrics_eval_all)
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path_log = f'{dir_evaluation_results}/eval_{name_baseline}_thres{eval_threshold}_{str_date}.txt'
        EvalScanNet.save_evaluation_results_to_latex(path_log, 
                                                        header = f'{name_baseline}\n                                     Accu.      Comp.      Prec.     Recall     F-score \n',
                                                        results = metrics_eval_all, 
                                                        names_item = exp_name_ls,
                                                        save_mean_num = 4,
                                                        mode = 'a')


    if args.mode == 'eval_mesh_2D_metrics':   # 实际上只比较了depth的准确度
        # 疑问：使用更少数据训练得到的mesh在测试时是按照所有图像标准进行测试吗？
        
        name_baseline = 'neus'
        eval_type_baseline = 'mesh'
        scale_depth = False

        # dir_results_baseline = f'./exps/evaluation/results_baselines/{name_baseline}'
        results_all =  []
        exp_name_ls = []

        for scene_name in scene_names:
            for exp_name in exp_dict[scene_name]:
                # scene_name += '_corner'
                print(f'Processing {scene_name}/{exp_name}...')
                exp_name_ls.append(f'{scene_name}/{exp_name}')
                dir_scan = f'{dir_dataset}/{scene_name}'
                if eval_type_baseline == 'mesh':
                    # use rendered depth map
                    # path_mesh_baseline =  f'{dir_results_baseline}/{scene_name}.ply'    # 这个path_mesh_baseline是指的预测的mesh吗？
                    path_mesh_pred = f'{dir_exp_results}/{scene_name}/{exp_name}/meshes/00160000_reso512_{scene_name}_world.ply'
                    # 是否需要像上面那样
                    pred_depths = render_depthmaps_pyrender(path_mesh_pred, path_intrin,
                                                                dir_poses=f'{dir_scan}/pose')
                    img_names = IOUtils.get_files_stem(f'{dir_scan}/depth', '.png')
                elif eval_type_baseline == 'depth':   # 这下面这个模式没有改
                    dir_depth_baseline =  f'{dir_exp_results}/{scene_name}'
                    pred_depths = GeoUtils.read_depth_maps_np(dir_depth_baseline)
                    img_names = IOUtils.get_files_stem(dir_depth_baseline, '.npy')

                # evaluation
                dir_gt_depth = f'{dir_scan}/depth'
                gt_depths, _ = EvalScanNet.load_gt_depths(img_names, dir_gt_depth)
                err_gt_depth_scale = EvalScanNet.depth_evaluation(gt_depths, pred_depths, dir_evaluation_results, scale_depth=scale_depth)
                results_all.append(err_gt_depth_scale)
            
        results_all = np.array(results_all)
        print('All results: ', results_all)

        count = 0
        str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")

        path_log_all = f'{dir_evaluation_results}/eval_{name_baseline}-scale_{scale_depth}_{eval_type_baseline}_{str_date}.txt'
        EvalScanNet.save_evaluation_results_to_latex(path_log_all, header = f'{str_date}\n\n',  mode = 'a')
        
        precision = 3
        results_all = np.round(results_all, decimals=precision)
        EvalScanNet.save_evaluation_results_to_latex(path_log_all, 
                                                        header = f'{name_baseline}\n                                     abs_rel.      sq_rel.      rmse.      rmse_log.     a1.      a2.      a3 \n',
                                                        results = results_all, 
                                                        names_item = exp_name_ls,
                                                        save_mean_num = 4,
                                                        mode = 'a',
                                                        precision = precision)

    """无normal的gt
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
        """

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
        


    if args.mode == 'eval_errormap':

        from matplotlib import cm
        import open3d as o3d

        def nn_correspondance(verts1, verts2):
        # copy自EvalScanNet, 对verts2中每一个vertex找到verts1中最近的vertex
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


        error_bound = 0.05
        eval_type_baseline = 'triangle_mesh'
        color_map = cm.get_cmap('Reds')

        for scene_name in scene_names:
            path_gt = f'{dir_dataset}/{scene_name}/{scene_name}_vh_clean_2_clean.ply'

            if eval_type_baseline == 'point_cloud':
                down_sample = 0.02
                pcd_gt = GeoUtils.read_point_cloud(path_gt)
                pcd_gt = pcd_gt.voxel_down_sample(down_sample)
                verts_gt = np.asarray(pcd_gt.points)
                for exp_name in exp_dict[scene_name]:
                    path_pred = f'{dir_exp_results}/{scene_name}/{exp_name}/meshes/00160000_reso512_{scene_name}_world_clean_bbox_faces_mask.ply'
                    pcd_pred = GeoUtils.read_point_cloud(path_pred)
                    pcd_pred = pcd_pred.voxel_down_sample(down_sample)
                    verts_pred = np.asarray(pcd_pred.points)

                    _, dist = nn_correspondance(verts_pred, verts_gt)
                    dist = np.array(dist)

                    dist_score = dist.clip(0, error_bound) / error_bound
                    colors = color_map(dist_score)[:, :3]

                    GeoUtils.save_points(f'{dir_evaluation_results}/errormap/{exp_name}_errormap_cloud_{error_bound}.ply', verts_gt, colors)

            elif eval_type_baseline == 'triangle_mesh':
                mesh_gt = GeoUtils.read_triangle_mesh(path_gt)
                verts_gt = np.asarray(mesh_gt.vertices)
                triangles_gt = np.asarray(mesh_gt.triangles)
                for exp_name in exp_dict[scene_name]:
                    path_pred = f'{dir_exp_results}/{scene_name}/{exp_name}/meshes/00160000_reso512_{scene_name}_world_clean_bbox_faces_mask.ply'
                    mesh_pred = GeoUtils.read_triangle_mesh(path_pred)
                    verts_pred = np.asarray(mesh_pred.vertices)
                    triangles_pred = np.asarray(mesh_pred.triangles)

                    _, dist = nn_correspondance(verts_pred, verts_gt)
                    dist = np.array(dist)

                    dist_score = dist.clip(0, error_bound) / error_bound
                    colors = color_map(dist_score)[:, :3]

                    GeoUtils.save_mesh(f'{dir_evaluation_results}/errormap/{exp_name}_errormap_mesh_{error_bound}.ply', verts_gt, triangles_gt, colors)
                    print('{}: {}'.format("complete", exp_name))


    print('Done')
