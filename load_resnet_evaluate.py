import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import ECGImageDataset
from resnet import *

from torchsummary import summary

model = ResNet50(4)
model.load_state_dict(torch.load('/mnt/2TBData/hemaxi/ProjetoDL/trained_models/resnet'))

#summary(model.cuda(), (9,256,256))

model.eval()


test_dataset = ECGImageDataset('/dev/shm/dataset', [17111,2156,2163], 'test')

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(evaluate(model, test_dataloader, 'test', gpu_id=None))
