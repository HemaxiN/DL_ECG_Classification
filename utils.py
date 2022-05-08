#code based on the source code of homework 1 and homework 2 of the 
#deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import cv2
import os
import tifffile

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

def plot_losses(epochs, valid_losses, train_losses, ylabel ='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    #plt.xticks(epochs)
    plt.plot(epochs, valid_losses, label='validation')
    plt.plot(epochs, train_losses, label='train')
    plt.legend()
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

#create a generator to read the images as we train the model
#(similar to flow_from_directory Keras)
class ECGImageDataset(Dataset):
    '''
    path/train/images
              /labels
        /val/images
            /labels
        /test/images
             /labels
    '''
    def __init__(self, path, train_dev_test, part='train'):
        self.path = path
        self.part = part
        self.train_dev_test = train_dev_test

    def __len__(self):
        if self.part == 'train':
            return self.train_dev_test[0]
        elif self.part == 'dev':
            return self.train_dev_test[1]
        elif self.part == 'test':
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        X, y = read_data(self.path, self.part, idx)
        return torch.tensor(X).float(), torch.tensor(y).float()


def read_data(path, partition, idx):
    '''Read the ECG Image Data'''
    final_path = os.path.join(path, partition)
    index = idx
    image = tifffile.imread(os.path.join(final_path, 'images/'+str(index)+'.tif'))
    image = image/255.0 #normalization
    #image = image.transpose(2,0,1) #channels, x, y
    label = np.load(os.path.join(final_path, 'labels/'+str(index)+'.npy'))
    return image, label


#performance evaluation, compute the tp, fn, fp, and tp for each disease class
#and compute the specificity and sensitivity
def compute_scores(y_true, y_pred, matrix):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        for i in range(0,4): #for each class
            matrix = computetpfnfp(pred[i], gt[i], i, matrix)
    return matrix


def compute_scores_dev(matrix):
    matrix[matrix==0] = 0.01
    #print(matrix)
    sensitivity = matrix[:,0] / (matrix[:,0] + matrix[:,1]) #tp/(tp+fn)
    specificity = matrix[:,3] / (matrix[:,3] + matrix[:,2]) #tn/(tn+fp)
    return np.mean(sensitivity), np.mean(specificity)

def computetpfnfp(pred, gt, i, matrix):
    if pred==0 and pred==0: #tn
        matrix[i,3] +=1
    if gt==1 and pred==0: #fn
        matrix[i,1] +=1
    if gt==0 and pred==1: #fp
        matrix[i,2] +=1
    if gt==1 and pred==1: #tp
        matrix[i,0] +=1
    return matrix
