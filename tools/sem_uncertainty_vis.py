from glob import glob
import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import utils.utils_nyu as utils_nyu

Data_dir = '../Data/dataset/indoor'
exps_dir = '../exps/indoor/neus/'
scene_list = ['scene0050_00']
semantic_type_list = ['semantic_pred']
method_name_list = ['con_test1', 'con_test2', 'con_test3','pred_40_test1', 'pred_40_test2_compa3', 'pred_40_test10_compa1', 'pred_40_test10_compa2']
colour_map_np = utils_nyu.nyu40_colour_code
for scene_name in scene_list:
    for semantic_type in semantic_type_list:
        # read semantic
        semantic_vis_dir = os.path.join(Data_dir, scene_name, f'{semantic_type}_vis')
        semantic_vis_lis = glob(f'{semantic_vis_dir}/*.png')
        semantic_vis_lis.sort(key=lambda x:int((x.split('/')[-1]).split('.')[0]))
        N_img = len(semantic_vis_lis)
        # read mv_similarity
        mv_dir = os.path.join(Data_dir, scene_name, 'mv_similarity', semantic_type)
        # read sem_uncertainty
        for method_name in method_name_list:
            sem_uncertainty_dir = os.path.join(exps_dir, scene_name, method_name, 'sem_uncertainty_vis')
            sem_uncertainty_list = sorted(glob(os.path.join(sem_uncertainty_dir, '*.png')))
            sem_render_dir = os.path.join(exps_dir, scene_name, method_name, 'semantic_npz')
            sem_render_list = sorted(glob(os.path.join(sem_render_dir, '*.npz')))

            save_dir = os.path.join(exps_dir, scene_name, method_name, 'uncertainty_mv_vis')
            os.makedirs(save_dir, exist_ok=True)

            sem_uncertainty = cv2.imread(sem_uncertainty_list[0])
            H, W, _=sem_uncertainty.shape
            video_name = os.path.join(save_dir, 'uncertainty.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
            videowriter = cv2.VideoWriter(video_name, fourcc, 10, (W*4+30, H))

            for idx in range(N_img):
                # 前端语义
                semantic_vis = cv2.imread(semantic_vis_lis[idx])
                # vis_mv
                mv_simialrity = np.load(os.path.join(mv_dir, f'{idx}.npz'))['arr_0']
                colormap_func = matplotlib.cm.get_cmap("jet")
                semantic_similarity_vis = colormap_func(mv_simialrity)[:, :, :3]
                semantic_similarity_vis = (semantic_similarity_vis*255).astype('uint8')
                # render语义
                sem_render = np.load(sem_render_list[idx])['arr_0']
                sem_render_vis = (colour_map_np[sem_render])[...,::-1]

                # vis uncertainty
                sem_uncertainty = cv2.imread(sem_uncertainty_list[idx])
                
                semantic_vis = cv2.resize(semantic_vis , (W, H))
                semantic_similarity_vis = cv2.resize(semantic_similarity_vis , (W, H))
                sem_render_vis = cv2.resize(sem_render_vis , (W, H))

                img_cat=(255 * np.ones((H, 10, 3))).astype('uint8')
                lis=[semantic_vis, img_cat, semantic_similarity_vis, img_cat, sem_render_vis, img_cat, sem_uncertainty]
                vis = np.concatenate(lis, axis=1)

                cv2.imwrite(os.path.join(save_dir, f'{idx}.png'), vis)
                videowriter.write(vis.astype(np.uint8))
            
            videowriter.release()




        