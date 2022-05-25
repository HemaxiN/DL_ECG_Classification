import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import configure_seed, configure_device, plot, ECGImageDataset, compute_scores_dev, compute_scores, plot_losses

#auxiliary functions to evaluate the performance of the model
from sklearn.metrics import recall_score
import statistics
import numpy as np

import os

import torch
import torch.nn as  nn
import torch.nn.functional as F

#code adapted from:
#https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=9):
        super(ResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=16)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=32, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=64, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

       
def ResNet50(num_classes, channels=9):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

def train_batch(X, y, model, optimizer, criterion, gpu_id=None, **kwargs):
    """
    X (batch_size, 9, 1000, 1000): batch of examples
    y (batch_size, 4): ground truth labels
    model: Pytorch model
    optimizer: optimizer for the gradient step
    criterion: loss function
    """
    X, y = X.to(gpu_id), y.to(gpu_id)
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(model, X):
    """
    Make label predictions for "X" (batch_size, 9, 1000, 1000) 
    given the trained model "model"
    """
    logits_ = model(X) # (batch_size, n_classes
    probabilities = torch.sigmoid(logits_).cpu()
    pred_labels = np.array(probabilities>0.5, dtype=float) # (batch_size, n_classes)
    return pred_labels

def evaluate(model,dataloader, part, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 9, 1000, 1000) : batch of examples
    y (batch_size,4): ground truth labels
    """
    model.eval()
    with torch.no_grad():
        matrix = np.zeros((4,4))
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = predict(model, x_batch)
            #print('true')
            y_true = np.array(y_batch.cpu())
            #print(y_true)
            #print('pred')
            #print(y_pred)
            matrix = compute_scores(y_true,y_pred, matrix)

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

    if part == 'dev':
        return compute_scores_dev(matrix)
    if part == 'test':
        return matrix

def compute_loss(model, dataloader, criterion, gpu_id=None):
    model.eval()
    with torch.no_grad():
        val_losses = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_losses.append(loss.item())
            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

        return statistics.mean(val_losses)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default=None,
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=100, type=int,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=4, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-path_save_model', default=None,
                        help='Path to save the model')
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    _examples_ = [17111,2156,2163]

    print("Loading data...") ## input manual nexamples train, dev e test
    train_dataset = ECGImageDataset(opt.data, _examples_, 'train')
    dev_dataset = ECGImageDataset(opt.data, _examples_, 'dev')
    test_dataset = ECGImageDataset(opt.data, _examples_, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)


    n_classes = 4  # 4 diseases + normal

    # initialize the model
    model = ResNet50(n_classes)
    model = model.to(opt.gpu_id)

    # get an optimizer
    optims = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion and compute the class weights (nbnegative/nbpositive)
    # according to the comments https://discuss.pytorch.org/t/weighted-binary-cross-entropy/51156/6
    # and https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573/2
    class_weights=torch.tensor([9528/5486, 9528/5250, 9528/4907, 9528/2655],dtype=torch.float)  
    class_weights = class_weights.to(opt.gpu_id)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) #https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_mean_losses = []
    valid_specificity = []
    valid_sensitivity = []
    train_losses = []
    last_val_loss = 10000
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            print('{} of {}'.format(i + 1, len(train_dataloader)), end='\r', flush=True)
            #print(i, flush=True)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion, gpu_id=opt.gpu_id)
            #input()
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            #input()
            train_losses.append(loss)
            #print(loss, flush=True)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        sensitivity, specificity = evaluate(model, dev_dataloader, 'dev', gpu_id=opt.gpu_id)
        val_loss = compute_loss(model, dev_dataloader, criterion, gpu_id=opt.gpu_id)
        valid_mean_losses.append(val_loss)
        valid_sensitivity.append(sensitivity)
        valid_specificity.append(specificity)
        print('Valid specificity: %.4f' % (valid_specificity[-1]))
        print('Valid sensitivity: %.4f' % (valid_sensitivity[-1]))
        
        if val_loss<last_val_loss:
            #https://pytorch.org/tutorials/beginner/saving_loading_models.html (save the model at the end of each epoch)
            torch.save(model.state_dict(), os.path.join(opt.path_save_model, 'model'+ str(ii.item())))
            last_val_loss = val_loss

    print('Final Test Results:')
    print(evaluate(model, test_dataloader, 'test', gpu_id=opt.gpu_id))
    # plot
    plot_losses(epochs, valid_mean_losses, train_mean_losses, ylabel='Loss', name='training-validation-loss-{}-{}'.format(opt.learning_rate, opt.optimizer))
    plot(epochs, valid_specificity, ylabel='Specificity', name='validation-specificity-{}-{}'.format(opt.learning_rate, opt.optimizer))
    plot(epochs, valid_sensitivity, ylabel='Sensitivity', name='validation-sensitivity-{}-{}'.format(opt.learning_rate, opt.optimizer))



if __name__ == '__main__':
    main()
