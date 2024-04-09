import os, logging,argparse
from datetime import datetime
import numpy as np
from glob import glob
import cv2

import evaluation.EvalScanNet as EvalScanNet
import utils.utils_semantic as SemanticUtils

from confs.path import lis_name_scenes

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

exp_name = 'deeplab40retrain'
semantic_class=40
num = 49

dir_results_baseline = '/home/du/Proj/3Dv_Reconstruction/Manhattan_sdf/exp/result/manhattan_sdf'
str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
dir_dataset = '../Data/dataset/indoor'
#--------------------------------------
# eval_3D_mesh_TSDF
eval_threshold_lis = [0.05]

metrics_eval_all = []
for scene_name in lis_name_scenes:
    logging.info(f'Processing: {scene_name}')
    dir = exp_name + '_' + scene_name[-7:-3]
    path_mesh_pred = f'{dir_results_baseline}/{dir}/{num}.ply'
    metrics_eval =  EvalScanNet.evaluate_3D_mesh_TSDF(path_mesh_pred,
                                                        scene_name,
                                                        dir_dataset = '../Data/dataset/indoor',
                                                        eval_threshold = eval_threshold_lis)
    
    metrics_eval_all.append(metrics_eval)
metrics_eval_all = np.array(metrics_eval_all)

path_log = f'{dir_results_baseline}/{exp_name}_eval3Dmesh_TSDF_{str_date}.md'

# markdown_header=f'| scene_name   |    Method|    Accu.|    Comp.|    Prec.|   Recall|  F-score|  Chamfer\n'
# markdown_header=markdown_header+'| -------------| ---------| ------- | ------- | ------- | ------- | ------- | ------- |\n'
# EvalScanNet.save_evaluation_results_to_markdown(path_log, 
#                                                 header = markdown_header, 
#                                                 name_baseline=exp_name,
#                                                 results = metrics_eval_all, 
#                                                 names_item = lis_name_scenes, 
#                                                 mode = 'w')

# #--------------------------------------
# # eval_semantic_2D
# metric_train_all = []
# for scene_name in lis_name_scenes:
#     logging.info(f'Processing {scene_name}...')
#     dir = exp_name + '_' + scene_name[-7:-3]

#     dir_scan = f'../Data/dataset/indoor/{scene_name}'
#     dir_exp = f'{dir_results_baseline}/{dir}/semantic'

#     semantic_GT_list=[]
#     semantic_render_list=[]
#     semantic_GT_lis = sorted(glob(f'{dir_scan}/semantic/train/semantic_GT/*.png'))

#     semantic_GT_list=[]
#     semantic_render_list=[]
#     for idx in range(len(semantic_GT_lis)):
#         semantic_GT=(cv2.imread(semantic_GT_lis[idx], cv2.IMREAD_UNCHANGED)).astype(np.uint8)
#         semantic_render=(cv2.imread(f'{dir_exp}/{idx}.png', cv2.IMREAD_UNCHANGED)).astype(np.uint8)
        
#         semantic_GT_copy = semantic_GT.copy()
#         semantic_render_copy = semantic_render.copy()
#         # merge语义
#         if semantic_class==3:
#             label_mapping_nyu=SemanticUtils.mapping_nyu3()
#         if semantic_class==40:
#             label_mapping_nyu=SemanticUtils.mapping_nyu40()
#         for scan_id, nyu_id in label_mapping_nyu.items():
#             semantic_GT_copy[semantic_GT==scan_id] = nyu_id
#             semantic_render_copy[semantic_render==scan_id] = nyu_id
            
#         semantic_GT_list.append(np.array(semantic_GT_copy))
#         semantic_render_list.append(semantic_render_copy)
    
#     if semantic_class>3:
#         true_labels=np.array(semantic_GT_list)-1
#         predicted_labels=np.array(semantic_render_list)-1
#     else:
#         true_labels=np.array(semantic_GT_list)
#         predicted_labels=np.array(semantic_render_list)  
#     metric_avg, exsiting_label, class_iou, class_accuray = SemanticUtils.compute_segmentation_metrics(true_labels=true_labels, 
#                                                                                         predicted_labels=predicted_labels, 
#                                                                                         semantic_class=semantic_class, 
#                                                                                         ignore_label=255)
#     metric_train_all.append(metric_avg)
#     path_log = f'{dir_exp}/{exp_name}_evalsemantic2D_{semantic_class}_{str_date}_markdown.md'
#     markdown_header_0=[f' {label} |'for label in exsiting_label]
#     markdown_header_1 = '\n| -------------| ---------|'+'---------|'*len(exsiting_label)
#     markdown_header=  'IoU\n| scene_ name   |   Method|'+''.join(markdown_header_0)+markdown_header_1+'\n'

#     EvalScanNet.save_evaluation_results_to_markdown(path_log, 
#                                                 header = markdown_header, 
#                                                 name_baseline=exp_name,
#                                                 results = [class_iou], 
#                                                 names_item = [scene_name],  
#                                                 save_mean=False,
#                                                 mode = 'w')

# path_log = f'{dir_results_baseline}/{exp_name}_evalsemantic2D_{semantic_class}_{str_date}_markdown.md'
# markdown_header='train\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
# markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
# EvalScanNet.save_evaluation_results_to_markdown(path_log, 
#                                                 header = markdown_header, 
#                                                 name_baseline=exp_name,
#                                                 results = metric_train_all, 
#                                                 names_item = lis_name_scenes,  
#                                                 mode = 'w')

# #--------------------------------------
# # eval_semantic_3D
# metrics_surface_all = []

# for scene_name in lis_name_scenes:
#     logging.info(f'Processing {scene_name}...')
#     dir = exp_name + '_' + scene_name[-7:-3]
    
#     dir_exp = f'{dir_results_baseline}/{dir}/semantic'
#     file_mesh_trgt = os.path.join(dir_dataset, f'{scene_name}/{scene_name}_vh_clean_2.labels.ply')
            
#     file_mesh_pred = f'{dir_results_baseline}/{dir}/{num}_sem.ply'
#     file_semseg_pred = f'{dir_results_baseline}/{dir}/semantic_surface.npz'
#     file_mesh_pred_transfer = f'{dir_results_baseline}/{dir}/{scene_name}_semantic_surface_transfer.ply'
    
#     mesh_transfer, metric_avg, exsiting_label, class_iou, class_accuray = SemanticUtils.evaluate_semantic_3D(file_mesh_trgt,
#                                                                                                             file_mesh_pred,
#                                                                                                             file_semseg_pred,
#                                                                                                             semantic_class=semantic_class)
#     mesh_transfer.export(file_mesh_pred_transfer)

#     metrics_surface_all.append(metric_avg)
#     path_log = f'{dir_exp}/{exp_name}_evalsemantic3D_{semantic_class}_{str_date}_markdown.md'
#     markdown_header_0=[f' {label} |'for label in exsiting_label]
#     markdown_header_1 = '\n| -------------| ---------|'+'---------|'*len(exsiting_label)
#     markdown_header=  'IoU\n| scene_ name   |   Method|'+''.join(markdown_header_0)+markdown_header_1+'\n'

#     EvalScanNet.save_evaluation_results_to_markdown(path_log, 
#                                                 header = markdown_header, 
#                                                 name_baseline=exp_name,
#                                                 results = [class_iou], 
#                                                 names_item = [scene_name],  
#                                                 save_mean=False,
#                                                 mode = 'w')

# path_log = f'{dir_results_baseline}/{exp_name}_evalsemantic3D_{semantic_class}_{str_date}_markdown.md'
# markdown_header='train\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
# markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
# EvalScanNet.save_evaluation_results_to_markdown(path_log, 
#                                                 header = markdown_header, 
#                                                 name_baseline=exp_name,
#                                                 results = metrics_surface_all, 
#                                                 names_item = lis_name_scenes,  
#                                                 mode = 'w')