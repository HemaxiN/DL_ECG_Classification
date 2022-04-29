import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import cv2
import os

def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


class ECGImageDataset(Dataset):
	'''
	path/train/images
			  /labels
		/val/images
			/labels
		/test/images
			 /labels
	'''
    def __init__(self, path, part='train'):
    	self.path = path
    	self.part = part

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
    	X, y = read_data(self.path, self.part, idx)
        return torch.tensor(X).float(), torch.tensor(self.y).long()


def read_data(path, partition, idx):
	'''Read the ECG Image Data'''
	final_path = os.path.join(path, partition)
	X = []
	y = []
	for index in idx:
		image = cv2.imread(os.path.join(final_path, 'images/'+str(index)+'.jpg'))
		image = image.transpose(2,0,1) #channels, x, y
		X.append(image)
		label = np.load(os.path.join(final_path, 'labels/'+str(index)+'.npy'))
		y.append(label)
	return X, y