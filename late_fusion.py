'''
This script deals with the late fusion approach for multimodal learning of ECG classification.
In this approach, predictions from unimodal approaches (1D signal and images) are fused and used as the inputs
on a new feedforward network (2 dense layers).

Code backbone of DSL homeworks was used to structure this script.
'''

import argparse

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import configure_device, configure_seed, ECGImageDataset, Dataset_for_RNN, plot, plot_losses
import gru as gru
import numpy as np

import AlexNet as alexnet
import resnet as resnet

from datetime import datetime
import os
from count_parameters import count_parameters


def modalities_evaluate(model, dataloader, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,4): ground truth labels_train
    """
    model.eval()
    with torch.no_grad():
        pred_logger = []
        y_logger = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = gru.predict(model, x_batch)
            y_true = np.array(y_batch.cpu())

            pred_logger.append(y_pred)
            y_logger.append(y_true)

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

    return pred_logger, y_logger


class LateFusionDataset(Dataset):
    def __init__(self, sig_path, img_path, sig_model, img_model, sig_type, img_type,
                 train_dev_test, gpu_id, batch_size, part='train'):
        self.sig_path = sig_path
        self.img_path = img_path
        self.sig_model = sig_model
        self.img_model = img_model
        self.sig_type = sig_type
        self.img_type = img_type
        self.part = part
        self.train_dev_test = train_dev_test
        self.gpu_id = gpu_id
        self.batch_size = batch_size

        self.sig_dataset = Dataset_for_RNN(self.sig_path, self.train_dev_test, self.part)
        self.img_dataset = ECGImageDataset(self.img_path, self.train_dev_test, self.part)

        self.sig_dataloader = DataLoader(self.sig_dataset, batch_size=self.batch_size, shuffle=False)
        self.img_dataloader = DataLoader(self.img_dataset, batch_size=self.batch_size, shuffle=False)

        self.X = []
        self.y = []

        self.compute_preds()

    def __len__(self):
        if self.part == 'train':
            return self.train_dev_test[0]
        elif self.part == 'dev':
            return self.train_dev_test[1]
        elif self.part == 'test':
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).float()

    def compute_preds(self):
        '''
        Loads data, performs prediction using the unimodal models and concatenates data.
        '''
        if self.sig_type == 'gru':
            sig_pred, sig_y = modalities_evaluate(self.sig_model, self.sig_dataloader, gpu_id=self.gpu_id)
        else:
            raise ValueError('1D model is not defined.')

        if self.img_type == 'alexnet':
            img_pred, img_y = modalities_evaluate(self.img_model, self.img_dataloader, gpu_id=self.gpu_id)
        elif self.img_type == 'resnet':
            img_pred, img_y = modalities_evaluate(self.img_model, self.img_dataloader, gpu_id=self.gpu_id)
        else:
            raise ValueError('2D model is not defined.')

        for i in range(len(sig_pred)):

            if len(self.X) == 0:
                self.X = np.c_[sig_pred[i], img_pred[i]]
                self.y = sig_y[i]
            else:
                self.X = np.r_[self.X, np.c_[sig_pred[i], img_pred[i]]]
                self.y = np.r_[self.y, sig_y[i]]


class LateFusionNet(nn.Module):
    def __init__(self, n_classes, n_features, hidden_size, dropout):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        activation_type (str)
        dropout (float): dropout probability
        """
        super(LateFusionNet, self).__init__()

        self.hidden = nn.Linear(n_features, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        x (batch_size x n_features): a batch of training examples
        """
        x = self.activation(self.hidden(x))
        x = self.dropout(x)
        x = self.out(x)

        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-signal_data', default='Dataset/data_for_rnn/', help="Path to the 1D ECG dataset.")
    parser.add_argument('-image_data', default='Dataset/Images/', help="Path to the 2D image dataset.")
    parser.add_argument('-signal_model', default='gru', help="Description of the 1D ECG model.")
    parser.add_argument('-image_model', default='alexnet', help="Description of the 2D image model.")
    parser.add_argument('-epochs', default=50, type=int, help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=64, type=int, help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.1)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-path_save_model', default='save_models/', help='Path to save the model')
    parser.add_argument('-hidden_size', type=int, default=16)
    opt = parser.parse_args()
    print(opt)

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    sig_type = opt.signal_model
    img_type = opt.image_model

    # LOAD MODELS
    if sig_type == 'gru':
        sig_path = 'best_trained_rnns/gru_3lay_128hu'
        hidden_size = 128
        num_layers = 3
        dropout_rate = 0.3

        sig_model = gru.RNN(3, hidden_size, num_layers, 4, dropout_rate, gpu_id=opt.gpu_id,
                            bidirectional=False).to(opt.gpu_id)

    else:
        raise ValueError('1D model is not defined.')

    if img_type == 'alexnet':
        img_path = 'Models/alexnet'
        img_model = alexnet.AlexNet(4).to(opt.gpu_id)

    elif img_type == 'resnet':
        img_path = 'Models/resnet'
        img_model = resnet.ResNet50(4).to(opt.gpu_id)

    else:
        raise ValueError('2D model is not defined.')

    sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(opt.gpu_id)))
    sig_model.eval()

    img_model.load_state_dict(torch.load(img_path, map_location=torch.device(opt.gpu_id)))
    img_model.eval()

    # LOAD DATA
    train_dataset = LateFusionDataset(opt.signal_data, opt.image_data, sig_model, img_model, sig_type, img_type,
                                      [17111, 2156, 2163], opt.gpu_id, opt.batch_size, part='train')
    dev_dataset = LateFusionDataset(opt.signal_data, opt.image_data, sig_model, img_model, sig_type, img_type,
                                    [17111, 2156, 2163], opt.gpu_id, opt.batch_size, part='dev')
    test_dataset = LateFusionDataset(opt.signal_data, opt.image_data, sig_model, img_model, sig_type, img_type,
                                     [17111, 2156, 2163], opt.gpu_id, opt.batch_size, part='test')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    model = LateFusionNet(4, 8, opt.hidden_size, opt.dropout).to(opt.gpu_id)

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
    class_weights = torch.tensor([9528 / 5486, 9528 / 5250, 9528 / 4907, 9528 / 2655], dtype=torch.float)
    class_weights = class_weights.to(opt.gpu_id)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights)  # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    count_parameters(model)

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_mean_losses = []
    valid_specificity = []
    valid_sensitivity = []
    train_losses = []
    for e in epochs:
        print('Training epoch {}'.format(e))
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            loss = gru.train_batch(
                X_batch, y_batch, model, optimizer, criterion, gpu_id=opt.gpu_id)
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        sensitivity, specificity = gru.evaluate(model, dev_dataloader, 'dev', gpu_id=opt.gpu_id)
        val_loss = gru.compute_loss(model, dev_dataloader, criterion, gpu_id=opt.gpu_id)
        valid_mean_losses.append(val_loss)
        valid_sensitivity.append(sensitivity)
        valid_specificity.append(specificity)
        print('Valid specificity: %.4f' % (valid_specificity[-1]))
        print('Valid sensitivity: %.4f' % (valid_sensitivity[-1]))

        dt = datetime.now()
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save the model at each epoch where the validation loss is the best so far
        if val_loss == np.min(valid_mean_losses):
            torch.save(model.state_dict(),
                       os.path.join(opt.path_save_model, str(int(datetime.timestamp(dt))) + 'late_model' + str(e.item())))

    # Results on test set:
    matrix = gru.evaluate(model, test_dataloader, 'test', gpu_id=opt.gpu_id)

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

    # plot
    plot_losses(epochs, valid_mean_losses, train_mean_losses, ylabel='Loss',
                name='training-validation-loss-late-{}-{}-{}-{}-{}-{}'.format(
             opt.learning_rate, opt.optimizer, opt.dropout, opt.epochs, opt.hidden_size, opt.batch_size))
    plot(epochs, valid_specificity, ylabel='Specificity',
         name='validation-specificity-late-{}-{}-{}-{}-{}-{}'.format(
             opt.learning_rate, opt.optimizer, opt.dropout, opt.epochs, opt.hidden_size, opt.batch_size))
    plot(epochs, valid_sensitivity, ylabel='Sensitivity',
         name='validation-sensitivity-late-{}-{}-{}-{}-{}-{}'.format(
             opt.learning_rate, opt.optimizer, opt.dropout, opt.epochs, opt.hidden_size, opt.batch_size))


if __name__ == '__main__':
    main()
