#import the necessary packages
import torch

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import ECGImageDataset
from AlexNet import *

#define the model
model = AlexNet(4)
#load the weighs of the model
model.load_state_dict(torch.load('/mnt/2TBData/hemaxi/ProjetoDL/model'))
model.eval() #model in the evaluation mode

#generator for the test dataset
test_dataset = ECGImageDataset('/dev/shm/dataset', [17111,2156,2163], 'test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#evaluate the performance of the model
print(evaluate(model, test_dataloader, 'test', gpu_id=None))