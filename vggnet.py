#Code based on the source code of homework 1 and homework 2 of the 
#deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks

#import the packages
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
from torch.nn import functional as F

#based on https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1 (visited on May 22, 2022)
class VGG16(nn.Module):
    def __init__(self, n_classes, **kwargs):
        super(VGG16, self).__init__()

        self.n_classes = n_classes

        n_filters = 16
        self.conv1_1 = nn.Conv2d(9, n_filters, 3, padding=1)
        self.conv1_2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)

        self.conv2_1 = nn.Conv2d(n_filters, n_filters*2, 3, padding=1)
        self.conv2_2 = nn.Conv2d(n_filters*2, n_filters*2, 3, padding=1)

        self.conv3_1 = nn.Conv2d(n_filters*2, n_filters*4, 3, padding=1)
        self.conv3_2 = nn.Conv2d(n_filters*4, n_filters*4, 3, padding=1)
        self.conv3_3 = nn.Conv2d(n_filters*4, n_filters*4,3, padding=1)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(65536, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.dropout(F.relu(self.conv2_2(x)),0.3)
        x = self.maxpool(x)
        x = F.dropout(x,0.3)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.dropout(F.relu(self.conv3_3(x)),0.3)
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x

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
            y_true = np.array(y_batch.cpu())
            matrix = compute_scores(y_true,y_pred, matrix)
            #delete unnecessary variables due to memory issues
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
    model = VGG16(n_classes)
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
    print('AAAAA')
    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_mean_losses = []
    valid_specificity = []
    valid_sensitivity = []
    train_losses = []
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
        
        if ii%20 ==0:
            #https://pytorch.org/tutorials/beginner/saving_loading_models.html (save the model at the end of each epoch)
            torch.save(model.state_dict(), os.path.join(opt.path_save_model, 'model'+ str(ii.item())))

    print('Final Test Results:')
    print(evaluate(model, test_dataloader, 'test', gpu_id=opt.gpu_id))
    # plot
    plot_losses(epochs, valid_mean_losses, train_mean_losses, ylabel='Loss', name='training-validation-loss-{}-{}'.format(opt.learning_rate, opt.optimizer))
    plot(epochs, valid_specificity, ylabel='Specificity', name='validation-specificity-{}-{}'.format(opt.learning_rate, opt.optimizer))
    plot(epochs, valid_sensitivity, ylabel='Sensitivity', name='validation-sensitivity-{}-{}'.format(opt.learning_rate, opt.optimizer))

if __name__ == '__main__':
    main()
