import torch
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from utils import ECGImageDataset
from alexnet import *

from torchsummary import summary

#define the model
model = AlexNet(4)
#load the weighs of the model
model.load_state_dict(torch.load('/mnt/2TBData/hemaxi/ProjetoDL/trained_models/alexnet'))
model.eval() #model in the evaluation mode

test_dataset = ECGImageDataset('/dev/shm/dataset', [17111,2156,2163], 'test')
dev_dataset = ECGImageDataset('/dev/shm/dataset', [17111,2156,2163], 'dev')

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=False)

# Threshold optimization on validation set
thr = threshold_optimization(model, dev_dataloader)

#thr = [0.4,0.4,0.4,0.4]

print('thresholds')
print(thr)

# Results on test set:
matrix = evaluate(model, test_dataloader, thr, gpu_id=None)

# compute sensitivity and specificity for each class:
MI_sensi = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
MI_spec = matrix[0, 3] / (matrix[0, 3] + matrix[0, 2])
STTC_sensi = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
STTC_spec = matrix[1, 3] / (matrix[1, 3] + matrix[1, 2])
CD_sensi = matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])
CD_spec = matrix[2, 3] / (matrix[2, 3] + matrix[2, 2])
HYP_sensi = matrix[3, 0] / (matrix[3, 0] + matrix[3, 1])
HYP_spec = matrix[3, 3] / (matrix[3, 3] + matrix[3, 2])

# compute mean sensitivity and specificity:
mean_sensi = np.mean(matrix[:, 0]) / (np.mean(matrix[:, 0]) + np.mean(matrix[:, 1]))
mean_spec = np.mean(matrix[:, 3]) / (np.mean(matrix[:, 3]) + np.mean(matrix[:, 2]))

# print results:
print('Final Test Results: \n ' + str(matrix) + '\n' + 'MI: sensitivity - ' + str(MI_sensi) + '; specificity - '
      + str(MI_spec) + '\n' + 'STTC: sensitivity - ' + str(STTC_sensi) + '; specificity - ' + str(STTC_spec)
      + '\n' + 'CD: sensitivity - ' + str(CD_sensi) + '; specificity - ' + str(CD_spec)
      + '\n' + 'HYP: sensitivity - ' + str(HYP_sensi) + '; specificity - ' + str(HYP_spec)
      + '\n' + 'mean: sensitivity - ' + str(mean_sensi) + '; specificity - ' + str(mean_spec))

