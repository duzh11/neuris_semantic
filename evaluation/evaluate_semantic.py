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

def evaluate_semantic(exp_name, lis_name_scenes,numclass,
                      dir_dataset,exp_dir,dir_results_baseline,
                      flag_old=False):    
    
    GT_name=f'semantic_{numclass}'
    metrics_average_all = []
    metrics_iou_all = []
    metrics_acc_all = []

    logging.info(f'Eval semantic class: {numclass}')
    for scene_name in lis_name_scenes:
        logging.info(f'eval scene: {scene_name}')
        #dir
        GT_dir=os.path.join(dir_dataset,scene_name,GT_name)

        if flag_old:
            render_dir=os.path.join(exp_dir,scene_name,exp_name,'semantic_render')
        else:
            render_dir=os.path.join(exp_dir,scene_name,exp_name,'semantic_npz')

        # render_dir=os.path.join(dir_dataset,scene_name,'semantic_deeplab')
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

        if numclass>3:
            true_labels=np.array(semantic_GT_list)-1
            predicted_labels=np.array(semantic_render_list)-1
        else:
            true_labels=np.array(semantic_GT_list)
            predicted_labels=np.array(semantic_render_list)  

        total_accuracy,class_average_accuracy,class_accuray, miou_valid_class,FW_iou,class_iou=SemanticUtils.calculate_segmentation_metrics(
                                                                                true_labels=true_labels, 
                                                                                predicted_labels=predicted_labels, 
                                                                                number_classes=numclass, 
                                                                                ignore_label=255)
        metrics_average=np.array([total_accuracy,class_average_accuracy,miou_valid_class,FW_iou])
        metrics_acc=np.array(class_accuray)
        metrics_iou=np.array(class_iou)
        metrics_average_all.append(metrics_average)
        metrics_acc_all.append(metrics_acc)
        metrics_iou_all.append(metrics_iou)

        logging.info(f'{scene_name}: {metrics_average}')

    metrics_eval_all = np.array(metrics_average_all)
    metrics_acc_all=np.array(metrics_acc_all)
    metrics_iou_all=np.array(metrics_iou_all)

    str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path_log = f'{dir_results_baseline}/{exp_name}_refuse/eval_{exp_name}_semantic_{str_date}_markdown.txt'

    markdown_header='Eval metrics\n| scene_ name   |   Method|  Acc.|  M_Acc|  M_IoU| FW_IoU|\n'
    markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
    EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                    header = markdown_header, 
                                                    exp_name=exp_name,
                                                    results = metrics_eval_all, 
                                                    names_item = lis_name_scenes, 
                                                    save_mean = True, 
                                                    mode = 'w')

    markdown_header='\nAccuracy\n'
    EvalScanNet.save_evaluation_results_to_txt(path_log, 
                                                    header = markdown_header, 
                                                    exp_name=exp_name,
                                                    results = metrics_acc_all, 
                                                    names_item = lis_name_scenes, 
                                                    save_mean = False, 
                                                    mode = 'a')
    markdown_header='\nIoU\n'
    EvalScanNet.save_evaluation_results_to_txt(path_log, 
                                                    header = markdown_header, 
                                                    exp_name=exp_name,
                                                    results = metrics_iou_all, 
                                                    names_item = lis_name_scenes, 
                                                    save_mean = False, 
                                                    mode = 'a') 

if __name__=='__main__':
    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)    
    lis_exp_name=['semantic_3']
    lis_name_scenes=['scene0084_00','scene0616_00']
    numclass=3
    
    dir_dataset='./Data/dataset/indoor'
    exp_dir='./exps/indoor/neus'
    dir_results_baseline = f'./exps/evaluation'
    
    for exp_name in lis_exp_name:
        logging.info(f'evaluate semantics: {exp_name}')
        evaluate_semantic(exp_name, lis_name_scenes, numclass,
                          dir_dataset, exp_dir, dir_results_baseline)