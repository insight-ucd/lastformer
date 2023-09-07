import numpy as np
import torch

###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class, 
                 cost_matrix=torch.tensor([[0.0, 10.0], [1.0, 0.0]]),
                 priors=torch.tensor([0.99, 0.01])):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
        self.ECn_binary = None  # Initialize binary expected cost to None
        self.cost_matrix = cost_matrix
        self.priors = priors

    def update_cm(self, pr, gt, weight=1):
        """Get current confusion matrix, compute current F1 score, and update the confusion matrix."""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def compute_ECn_binary(self, cost_matrix=None, priors=None):
        """Compute the ECn_binary from the confusion matrix."""
        assert self.n_class == 2, "Function only supports binary classification"
        if cost_matrix is None:
            cost_matrix = self.cost_matrix
        if priors is None:
            priors = self.priors

        TN, FP, FN, TP = self.sum[0, 0], self.sum[0, 1], self.sum[1, 0], self.sum[1, 1]
        
        R12 = FP / (FP + TN + np.finfo(np.float32).eps)
        R21 = FN / (FN + TP + np.finfo(np.float32).eps)
        
        # Compute alpha
        alpha = cost_matrix[0, 1].item() * priors[0].item() / (cost_matrix[1, 0].item() * priors[1].item() + np.finfo(np.float32).eps)
        
        # Compute ECn for binary classification using the provided formula
        if alpha >= 1:
            self.ECn_binary = alpha * R12 + R21
        else:
            self.ECn_binary = R12 + (1/alpha) * R21
        return self.ECn_binary

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        if self.n_class == 2 and self.ECn_binary is None:
            self.compute_ECn_binary()
        if self.n_class == 2:
            scores_dict['ECn_binary'] = self.ECn_binary
        return scores_dict



def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # Class-specific metrics
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))
    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    # Overall metrics
    overall_precision = tp.sum() / (tp.sum() + (sum_a0 - tp).sum())
    overall_recall = tp.sum() / (tp.sum() + (sum_a1 - tp).sum())
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    overall_iou = tp.sum() / (tp.sum() + (sum_a0 + sum_a1 - tp).sum())

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1,
                  'overall_precision': overall_precision,
                  'overall_recall': overall_recall,
                  'overall_f1': overall_f1,
                  'overall_iou': overall_iou}

    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)

    return score_dict



def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']

def compute_binary_expected_cost(predictions, targets, cost_matrix, priors):
    """
    Compute the Normalized Expected Cost (ECn) for binary classification.
    
    Args:
    - predictions (torch.Tensor): Predictions from the model. Shape (batch_size, 2).
    - targets (torch.Tensor): True labels. Shape (batch_size, ).
    - cost_matrix (torch.Tensor): Cost matrix. Shape (2, 2).
    - priors (torch.Tensor): Prior probabilities for each class. Shape (2, ).
    
    Returns:
    - float: The computed ECn for binary classification.
    """
    assert predictions.size(1) == 2, "Function only supports binary classification"
    
    # Compute R values
    R12 = ((targets == 0) & (predictions.argmax(dim=1) == 1)).float().mean()
    R21 = ((targets == 1) & (predictions.argmax(dim=1) == 0)).float().mean()
    
    # Compute alpha
    alpha = cost_matrix[0, 1] * priors[0] / (cost_matrix[1, 0] * priors[1])
    
    # Compute ECn for binary classification using the provided formula
    if alpha >= 1:
        ECn_binary = alpha * R12 + R21
    else:
        ECn_binary = R12 + (1/alpha) * R21
    
    return ECn_binary.item()
