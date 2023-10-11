import cv2, os, sys
sys.path.append(os.getcwd())
import numpy as np

from datetime import datetime
from glob import glob

import evaluation.EvalScanNet as EvalScanNet
import utils.utils_semantic as SemanticUtils

lis_name_scenes=['scene0378_00', 'scene0435_02']
semantic_type_lis = ['deeplab']
data_mode_lis = ['train', 'test']

dir_dataset = '../Data/dataset/indoor'
semantic_class = 40
MANHATTAN=False
str_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
for semantic_type in semantic_type_lis:
    print(f'***Evaluate semantic: {semantic_type}...\n\n')
    metric_train_all = []
    metric_test_all = []

    for scene_name in lis_name_scenes:
        dir_scan = f'{dir_dataset}/{scene_name}'
        print(f'Processing {scene_name}...')
        
        for data_mode in data_mode_lis:
            print(f'Evaluate train/test: {data_mode}')
            semantic_GT_lis = sorted(glob(f'{dir_scan}/semantic/{data_mode}/semantic_GT/*.png'))
            semantic_predicted_lis = sorted(glob(f'{dir_scan}/semantic/{data_mode}/{semantic_type}/*.png'))
            semantic_GT_lis.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            semantic_predicted_lis.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            
            semantic_GT_list=[]
            semantic_predicted_list=[]
            for idx in range(len(semantic_GT_lis)):
                semantic_GT=(cv2.imread(semantic_GT_lis[idx], cv2.IMREAD_UNCHANGED)).astype(np.uint8)
                semantic_predicted=(cv2.imread(semantic_predicted_lis[idx], cv2.IMREAD_UNCHANGED)).astype(np.uint8)

                reso=semantic_GT.shape[0]/semantic_predicted.shape[0]
                if reso>1:
                    semantic_GT=cv2.resize(semantic_GT, (semantic_predicted.shape[1],semantic_predicted.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                semantic_GT_copy = semantic_GT.copy()
                semantic_predicted_copy = semantic_predicted.copy()
                # 遵循Manhattan-sdf的语义merge策略
                if MANHATTAN:
                    if semantic_class==3:
                        label_mapping_nyu=SemanticUtils.mapping_nyu3(manhattan=MANHATTAN)
                    if semantic_class==40:
                        label_mapping_nyu=SemanticUtils.mapping_nyu40(manhattan=MANHATTAN)
                    for scan_id, nyu_id in label_mapping_nyu.items():
                        semantic_GT_copy[semantic_GT==scan_id] = nyu_id
                        semantic_predicted_copy[semantic_predicted==scan_id] = nyu_id
                
                semantic_GT_list.append(np.array(semantic_GT_copy))
                semantic_predicted_list.append(semantic_predicted_copy)

            if semantic_class>3:
                true_labels=np.array(semantic_GT_list)-1
                predicted_labels=np.array(semantic_predicted_list)-1
            else:
                true_labels=np.array(semantic_GT_list)
                predicted_labels=np.array(semantic_predicted_list)  

            metric_avg, exsiting_label, class_iou, class_accuray = SemanticUtils.compute_segmentation_metrics(true_labels=true_labels, 
                                                                                                predicted_labels=predicted_labels, 
                                                                                                semantic_class=semantic_class, 
                                                                                                ignore_label=255)
            if data_mode == 'train':
                metric_train_all.append(metric_avg)
            elif data_mode == 'test':
                metric_test_all.append(metric_avg)

            path_log = f'{dir_scan}/semantic/{semantic_type}_evalsemantic_{data_mode}_{semantic_class}_{MANHATTAN}_{str_date}_markdown.md'
            markdown_header_0=[f' {label} |'for label in exsiting_label]
            markdown_header_1 = '\n| -------------| ---------|'+'---------|'*len(exsiting_label)
            markdown_header=  'IoU\n| scene_ name   |   Method|'+''.join(markdown_header_0)+markdown_header_1+'\n'

            EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=semantic_type,
                                                        results = [class_iou], 
                                                        names_item = [scene_name],  
                                                        save_mean=False,
                                                        mode = 'w')
            
            markdown_header = '\nAcc\n| scene_ name   |   Method|' + ''.join(markdown_header_0)+markdown_header_1+'\n'
            EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                        header = markdown_header, 
                                                        name_baseline=semantic_type,
                                                        results = [class_accuray], 
                                                        names_item = [scene_name],  
                                                        save_mean=False,
                                                        mode = 'a')

    path_log = f'{os.path.dirname(dir_scan)}/{semantic_type}_evalsemantic_{semantic_class}_{str_date}_markdown.md'
    markdown_header='Train\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
    markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
    EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                    header = markdown_header, 
                                                    name_baseline=semantic_type,
                                                    results = metric_train_all, 
                                                    names_item = lis_name_scenes,  
                                                    mode = 'w')
    
    markdown_header='\nTest\n| scene_ name   |   Method|  Acc|  M_Acc|  M_IoU| FW_IoU|\n'
    markdown_header=markdown_header+'| -------------| ---------| ----- | ----- | ----- | ----- |\n'
    EvalScanNet.save_evaluation_results_to_markdown(path_log, 
                                                    header = markdown_header, 
                                                    name_baseline=semantic_type,
                                                    results = metric_test_all, 
                                                    names_item = lis_name_scenes,  
                                                    mode = 'a')
    

            

