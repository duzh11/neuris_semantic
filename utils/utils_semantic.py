import numpy as np

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