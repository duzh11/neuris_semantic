import numpy as np
import logging,os
import cv2
from sklearn.metrics import confusion_matrix

def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()

def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, ignore_label):
    true_labels=np.array(true_labels)
    predicted_labels=np.array(predicted_labels)
    
    if (true_labels == ignore_label).all():
        return [0]*4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    valid_pix_ids = true_labels!=ignore_label
    predicted_labels = predicted_labels[valid_pix_ids] 
    true_labels = true_labels[valid_pix_ids]
    
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(0, number_classes)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    average_accuracy = nanmean(np.diagonal(norm_conf_mat)) #平均精度
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)) #总精度
    class_accuray=np.diagonal(norm_conf_mat).copy() #类别精度
    # class_accuray[missing_class_mask]=-1
    
    class_iou = np.zeros(number_classes)

    for class_id in range(number_classes):
        class_iou[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id])) 
    average_iou = np.mean(class_iou[exsiting_class_mask]) #平均IoU
    
    #iou
    freq = conf_mat.sum(axis=1) / conf_mat.sum()
    FW_iou = (freq[freq > 0] * class_iou[freq > 0]).sum()

    return average_accuracy, total_accuracy, class_accuray, average_iou, FW_iou, class_iou 

def evaluate_semantic(exp_name, 
                      lis_name_scenes,
                      numclass,
                      dir_dataset='../Data/dataset/indoor',
                      exp_dir='../exps/indoor/neus',
                      flag=''):    
    
    GT_name=f'semantic_{numclass}'
    metrics_average = []
    metrics_iou = []
    metrics_acc = []

    logging.info(f'Eval semantic class: {numclass}')
    for scene_name in lis_name_scenes:
        logging.info(f'\n\nProcess semantic: {scene_name}')
        #dir
        GT_dir=os.path.join(dir_dataset,scene_name,GT_name)

        if flag=='old':
            render_dir=os.path.join(exp_dir,scene_name,exp_name,'semantic_render')
        else:
            render_dir=os.path.join(exp_dir,scene_name,exp_name,'semantic_npz')

        # render_dir=os.path.join(dir_dataset,scene_name,'semantic_deeplab')
        GT_list=os.listdir(GT_dir)
        id_list=[int(os.path.splitext(frame)[0]) for frame in GT_list]
        id_list=sorted(id_list)
        semantic_GT_list=[]
        semantic_render_list=[]
        
        for idx in id_list:
            GT_file=os.path.join(GT_dir, '%d.png'%idx)
            
            if flag=='old':
                render_file=os.path.join(render_dir, '00160000_'+'0'*(4-len(str(idx)))+str(idx)+'_reso2.png')
                # render_file=os.path.join(render_dir, str(idx)+'.png')#deeplab
            else:
                render_file=os.path.join(render_dir, '00160000_'+'0'*(4-len(str(idx)))+str(idx)+'_reso2.npz')

            semantic_GT=cv2.imread(GT_file)[:,:,0]
            
            if flag=='old':
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

        average_accuracy, total_accuracy, class_accuray, average_iou, FW_iou, class_iou=calculate_segmentation_metrics(
                                                                                true_labels=true_labels, 
                                                                                predicted_labels=predicted_labels, 
                                                                                number_classes=numclass, 
                                                                                ignore_label=255)
        
        metrics_average.append([average_accuracy, total_accuracy, average_iou, FW_iou])
        metrics_acc.append(class_accuray)
        metrics_iou.append(class_iou)
        # logging.info(f'{scene_name}: {[average_accuracy, total_accuracy, average_iou, FW_iou]}')

    return metrics_average, metrics_acc, metrics_iou