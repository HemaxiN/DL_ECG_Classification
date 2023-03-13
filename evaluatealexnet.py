import torch
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from utils import ECGImageDataset
from alexnet import *
from torchsummary import summary

model = AlexNet(4)
model.load_state_dict(torch.load('/mnt/2TBData/hemaxi/ProjetoDL/alexnet1/model162', map_location="cpu"))

gpu_id = None

model.eval()

test_dataset = ECGImageDataset('/dev/shm/dataset', [17111,2156,2163], 'test')
dev_dataset = ECGImageDataset('/dev/shm/dataset', [17111,2156,2163], 'dev')

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
dev_dataloader = DataLoader(dev_dataset, batch_size=2156, shuffle=False)

# Threshold optimization on validation set
thr = threshold_optimization(model, dev_dataloader, gpu_id=None)
print('thresholds')
print(thr)

# Results on test set:
matrix = evaluate(model, test_dataloader, thr, gpu_id=None)
aurocs = eval_auroc(model, test_dataloader, gpu_id=None)


print('aurocs: {}'.format(aurocs))

MI_sensi = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
MI_spec = matrix[0, 3] / (matrix[0, 3] + matrix[0, 2])
MI_acc = (matrix[0, 0] + matrix[0, 3]) / np.sum(matrix[0])
MI_prec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 2])
MI_f1 = (2 * matrix[0, 0]) / (2 * matrix[0, 0] + matrix[0, 2] + matrix[0, 1])
MI_auroc = aurocs[0]


STTC_sensi = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
STTC_spec = matrix[1, 3] / (matrix[1, 3] + matrix[1, 2])
STTC_acc = (matrix[1, 0] + matrix[1, 3]) / np.sum(matrix[1])
STTC_prec = matrix[1, 0] / (matrix[1, 0] + matrix[1, 2])
STTC_f1 = (2 * matrix[1, 0]) / (2 * matrix[1, 0] + matrix[1, 2] + matrix[1, 1])
STTC_auroc = aurocs[1]

CD_sensi = matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])
CD_spec = matrix[2, 3] / (matrix[2, 3] + matrix[2, 2])
CD_acc = (matrix[2, 0] + matrix[2, 3]) / np.sum(matrix[2])
CD_prec = matrix[2, 0] / (matrix[2, 0] + matrix[2, 2])
CD_f1 = (2 * matrix[2, 0]) / (2 * matrix[2, 0] + matrix[2, 2] + matrix[2, 1])
CD_auroc = aurocs[2]

HYP_sensi = matrix[3, 0] / (matrix[3, 0] + matrix[3, 1])
HYP_spec = matrix[3, 3] / (matrix[3, 3] + matrix[3, 2])
HYP_acc = (matrix[3, 0] + matrix[3, 3]) / np.sum(matrix[3])
HYP_prec = matrix[3, 0] / (matrix[3, 0] + matrix[3, 2])
HYP_f1 = (2 * matrix[3, 0]) / (2 * matrix[3, 0] + matrix[3, 2] + matrix[3, 1])
HYP_auroc = aurocs[3]

# compute mean sensitivity and specificity:
mean_mat = np.mean(matrix, axis=0)
mean_sensi = mean_mat[0] / (mean_mat[0] + mean_mat[1])
mean_spec = mean_mat[3] / (mean_mat[3] + mean_mat[2])
mean_acc = (mean_mat[0] + mean_mat[3]) / np.sum(mean_mat)
mean_prec = mean_mat[0] / (mean_mat[0] + mean_mat[2])
mean_f1 = (2 * mean_mat[0]) / (2 * mean_mat[0] + mean_mat[2] + mean_mat[1])
mean_auroc = aurocs.mean()
mean_g = np.sqrt(mean_spec * mean_sensi)


print('Final Test Results: \n ' + str(matrix) + '\n\n' +
      'MI: \n\tsensitivity - ' + str(MI_sensi) + '\n\tspecificity - ' + str(MI_spec) + '\n\tprecision - ' + str(MI_prec) + '\n\taccuracy - ' + str(MI_acc) + '\n\tF1 Score - ' + str(MI_f1) + '\n\tAUROC - ' + str(MI_auroc) + '\n' +
      'STTC: \n\tsensitivity - ' + str(STTC_sensi) + '\n\tspecificity - ' + str(STTC_spec) + '\n\tprecision - ' + str(STTC_prec) + '\n\taccuracy - ' + str(STTC_acc) + '\n\tF1 Score - ' + str(STTC_f1) + '\n\tAUROC - ' + str(STTC_auroc) + '\n' +
      'CD: \n\tsensitivity - ' + str(CD_sensi) + '\n\tspecificity - ' + str(CD_spec) + '\n\tprecision - ' + str(CD_prec) + '\n\taccuracy - ' + str(CD_acc) + '\n\tF1 Score - ' + str(CD_f1) + '\n\tAUROC - ' + str(CD_auroc) + '\n' +
      'HYP: \n\tsensitivity - ' + str(HYP_sensi) + '\n\tspecificity - ' + str(HYP_spec) + '\n\tprecision - ' + str(HYP_prec) + '\n\taccuracy - ' + str(HYP_acc) + '\n\tF1 Score - ' + str(HYP_f1) + '\n\tAUROC - ' + str(HYP_auroc) + '\n' +
      'mean: \n\tG-Mean - ' + str(mean_g) + '\n\tsensitivity - ' + str(mean_sensi) + '\n\tspecificity - ' + str(mean_spec) + '\n\tprecision - ' + str(mean_prec) + '\n\taccuracy - ' + str(mean_acc) + '\n\tF1 Score - ' + str(mean_f1) + '\n\tAUROC - ' + str(mean_auroc))