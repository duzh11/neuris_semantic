import numpy as np
from sklearn.metrics import confusion_matrix

#得到混淆矩阵
def ConfusionMatrix(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

#计算图像分割衡量系数
def eval_semantic(label_trues, label_preds, n_class):
    """
     :param label_preds: numpy data, shape:[batch,h,w]
     :param label_trues:同上
     :param n_class:类别数

     Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    label_trues=np.array(label_trues)
    label_preds=np.array(label_preds)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues,label_preds):
        hist += ConfusionMatrix(lt.flatten(), lp.flatten(), n_class)
    #平均精度
    acc_mean = np.diag(hist).sum() / hist.sum()
    #类别精度
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    #平均类别精度
    acc_cls_mean = np.nanmean(acc_cls)
    #iou
    iou = np.diag(hist) / ( hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) )
    #平均iou
    iou_mean = np.nanmean(iou)
    #频权交并比iou
    freq = hist.sum(axis=1) / hist.sum()
    iou_freq_mean = (freq[freq > 0] * iou[freq > 0]).sum()
    return acc_mean,acc_cls,acc_cls_mean, iou,iou_mean,iou_freq_mean

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

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat)) #平均类别精度
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)) #平均精度
    class_accuray=np.diagonal(norm_conf_mat).copy() #类别精度
    # class_accuray[missing_class_mask]=-1
    
    class_iou = np.zeros(number_classes)

    for class_id in range(number_classes):
        class_iou[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id])) 
    miou_valid_class = np.mean(class_iou[exsiting_class_mask]) #平均IoU
    
    #iou
    freq = conf_mat.sum(axis=1) / conf_mat.sum()
    FW_iou = (freq[freq > 0] * class_iou[freq > 0]).sum()

    return total_accuracy, class_average_accuracy, class_accuray, miou_valid_class,FW_iou, class_iou 