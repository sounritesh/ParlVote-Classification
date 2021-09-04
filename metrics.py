import torch
import numpy as np

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = acc * 100
    
    return acc

def binary_mcc(y_pred, y_test):
  with torch.no_grad():
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    confusion_vector = y_pred_tag / y_test
    TP = torch.sum(confusion_vector == 1).item()
    FP = torch.sum(confusion_vector == float('inf')).item()
    TN = torch.sum(torch.isnan(confusion_vector)).item()
    FN = torch.sum(confusion_vector == 0).item()

    t_sum = torch.tensor([TP+FP, TN+FN])
    p_sum = torch.tensor([TP+FN, TN+FP])
    n_correct = TP+TN
    n_samples = torch.sum(p_sum)

    cov_ytyp = n_correct * n_samples - torch.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - torch.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - torch.dot(t_sum, t_sum)
    mcc = cov_ytyp / (cov_ytyt * cov_ypyp)**0.5

    if np.isnan(mcc):
        return 0.
    else:
        return mcc