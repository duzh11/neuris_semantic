import os, logging
import numpy as np
import cv2
from datetime import datetime

import evaluation.EvalScanNet as EvalScanNet
import utils.utils_semantic as SemanticUtils

if __name__=='__main__':
    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)    
    lis_name_scenes=['scene0050_00','scene0616_00']
    lis_numclass=[40]
    
    dir_dataset='../Data/dataset/indoor'

    for scene_name in lis_name_scenes: 
        for numclass in lis_numclass:
            method_name='oneformer'
            metrics_average=[]
            metric_iou=[]
            metirc_accuracy=[]
            logging.info(f'\n\nProcess semantic: {scene_name}, semantic_class: {numclass}')   
            render_dir=os.path.join(dir_dataset,scene_name, method_name)
            # render_dir='/home/du/Proj/Geometry_3Dv/Manhattan_sdf/exp/result/manhattan_sdf/test9_0616_a/semantic'
            GT_name=f'semantic_GT'
            GT_dir=os.path.join(dir_dataset,scene_name,GT_name)
            GT_list=os.listdir(GT_dir)
            id_list=[int(os.path.splitext(frame)[0]) for frame in GT_list]
            id_list=sorted(id_list)
            semantic_GT_list=[]
            semantic_render_list=[]
            
            i=-1
            for idx in id_list:
                i+=1
                GT_file=os.path.join(GT_dir, '%d.png'%idx)
                render_file=os.path.join(render_dir, f'{idx}.png')
                # render_file=os.path.join(render_dir, f'{i}.png')

                semantic_GT=cv2.imread(GT_file)[:,:,0]
                semantic_render=cv2.imread(render_file)[:,:,0]
                
                semantic_seg=semantic_render.copy()
                if numclass==3:
                    label_mapping_nyu=SemanticUtils.mapping_nyu3(manhattan=True)
                if numclass==40:   
                    label_mapping_nyu=SemanticUtils.mapping_nyu40(manhattan=True)
                for scan_id, nyu_id in label_mapping_nyu.items():
                    semantic_seg[semantic_render==scan_id] = nyu_id
                
                semantic_GT_list.append(semantic_GT)
                semantic_render_list.append(semantic_seg)
            
            if numclass>3:
                true_labels=np.array(semantic_GT_list)-1
                predicted_labels=np.array(semantic_render_list)-1
            else:
                true_labels=np.array(semantic_GT_list)
                predicted_labels=np.array(semantic_render_list) 
            
            average_accuracy, total_accuracy, class_accuray, average_iou, FW_iou, class_iou=SemanticUtils.calculate_segmentation_metrics(
                                                                                true_labels=true_labels, 
                                                                                predicted_labels=predicted_labels, 
                                                                                number_classes=numclass, 
                                                                                ignore_label=255)
        
            metrics_average.append([average_accuracy, total_accuracy, average_iou, FW_iou])
            metric_iou.append(class_iou)
            metirc_accuracy.append(class_accuray)

            str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
            path_log = f'{scene_name}_{method_name}_{numclass}_{str_date}_markdown.txt'
            # path_log=os.path.join(dir_dataset,scene_name, method_name, path_log)
            path_log=os.path.join(render_dir, path_log)
            markdown_header='Eval metrics\n| scene_ name   |   Method|  Acc.|  M_Acc|  M_IoU| FW_IoU|\n'
            markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
            EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        exp_name=f'{method_name}_{numclass}',
                                                        results = metrics_average, 
                                                        names_item = [scene_name], 
                                                        save_mean = False, 
                                                        mode = 'w')
            
            EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = '\naccuracy\n', 
                                                        exp_name=f'{method_name}_{numclass}',
                                                        results = metirc_accuracy, 
                                                        names_item = [scene_name], 
                                                        save_mean = False, 
                                                        mode = 'a')
            
            EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = '\niou\n', 
                                                        exp_name=f'{method_name}_{numclass}',
                                                        results = metric_iou, 
                                                        names_item = [scene_name], 
                                                        save_mean = False, 
                                                        mode = 'a')
            

