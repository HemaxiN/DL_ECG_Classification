import torch

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import ECGImageDataset
from AlexNet import *

model = AlexNet(4)
model.load_state_dict(torch.load('/mnt/2TBData/hemaxi/DL/projeto/model'))
model.eval()


test_dataset = ECGImageDataset('/mnt/2TBData/hemaxi/DL/projeto/ds', [17111,2156,2163], 'test')

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(evaluate(model, test_dataloader, 'test', gpu_id=None))