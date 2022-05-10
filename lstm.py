import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import configure_seed, configure_device, plot, compute_scores_dev, compute_scores, Dataset_for_RNN

from datetime import datetime

# auxiliary functions to evaluate the performance of the model
from sklearn.metrics import recall_score
import statistics
import numpy as np

import os


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout_rate, gpu_id=None, **kwargs):
        """
        Define the layers of the model
        Args:
            input_size (int): "Feature" size (in this case, it is 3)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of hidden RNN layers
            n_classes (int): Number of classes in our classification problem
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.gpu_id = gpu_id
        self.dropout_rate = dropout_rate

        # RNN can be replaced with GRU/LSTM (for GRU the rest of the model stays exactly the same)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)  # batch_first means that the input must have as first dimension the batch size
        # x - > (batch_size, seq_length, input_size) (input of the model)

        self.fc = nn.Linear(hidden_size, n_classes)  # linear layer for the classification part
        # the fully connected layer (fc) only uses the last timestep of the output of the RNN to do the classification

        #self.ol = nn.Sigmoid()

    def forward(self, X, **kwargs):
        """
        Forward Propagation

        Args:
            X: batch of training examples with dimension (batch_size, 1000, 3)
        """
        # initial hidden state:
        h_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.gpu_id)
        c_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.gpu_id)

        out_rnn, _ = self.lstm(X, (h_0, c_0))
        # out_rnn shape: (batch_size, seq_length, hidden_size) = (batch_size, 1000, hidden_size)

        # decode the hidden state of only the last timestep (other approaches are possible, such as the mean of all states, ..)
        out_rnn = out_rnn[:, -1, :]
        # out_rnn shape: (batch_size, hidden_size) - ready to enter the fc layer

        out_fc = self.fc(out_rnn)
        # out_fc shape: (batch_size, num_classes)

        #out = self.ol(out_fc)
        # out shape: (batch_size, num_classes)

        return out_fc


def train_batch(X, y, model, optimizer, criterion, gpu_id=None, **kwargs):
    """
    X (batch_size, 1000, 3): batch of examples
    y (batch_size, 4): ground truth labels_train
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
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    logits_ = model(X)  # (batch_size, n_classes)
    probabilities = torch.sigmoid(logits_).cpu()
    pred_labels = np.array(probabilities > 0.5, dtype=float)  # (batch_size, n_classes)
    return pred_labels


def evaluate(model, dataloader, part, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,4): ground truth labels_train
    """
    model.eval()
    with torch.no_grad():
        matrix = np.zeros((4, 4))
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = predict(model, x_batch)
            y_true = np.array(y_batch.cpu())
            matrix = compute_scores(y_true, y_pred, matrix)

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

    if part == 'dev':
        return compute_scores_dev(matrix)
    if part == 'test':
        return matrix
        # cols: TP, FN, FP, TN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='Dataset/data_for_rnn/',
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=40, type=int,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=512, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-path_save_model', default='save_models/',
                        help='Path to save the model')
    parser.add_argument('-num_layers', type=int, default=2)
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    samples = [17111, 2156, 2163]
    print("Loading data...")  # input manual nexamples train, dev e test
    train_dataset = Dataset_for_RNN(opt.data, samples, 'train')
    dev_dataset = Dataset_for_RNN(opt.data, samples, 'dev')
    test_dataset = Dataset_for_RNN(opt.data, samples, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    input_size = 3
    hidden_size = 128
    num_layers = opt.num_layers
    n_classes = 4  # 4 diseases + normal

    # initialize the model
    model = LSTM(input_size, hidden_size, num_layers, n_classes, dropout_rate=opt.dropout, gpu_id=opt.gpu_id)
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
    class_weights = torch.tensor([9528 / 5486, 9528 / 5250, 9528 / 4907, 9528 / 2655], dtype=torch.float)
    class_weights = class_weights.to(opt.gpu_id)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights)  # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_specificity = []
    valid_sensitivity = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            # print('{} of {}'.format(i + 1, len(train_dataloader)), end='\r', flush=True)
            #print(i, flush=True)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion, gpu_id=opt.gpu_id)
            # input()
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            # input()
            train_losses.append(loss)
            # print(loss, flush=True)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        sensitivity, specificity = evaluate(model, dev_dataloader, 'dev', gpu_id=opt.gpu_id)
        valid_sensitivity.append(sensitivity)
        valid_specificity.append(specificity)
        print('Valid specificity: %.4f' % (valid_specificity[-1]))
        print('Valid sensitivity: %.4f' % (valid_sensitivity[-1]))

        dt = datetime.now()
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html (save the model at the end of each epoch)
        torch.save(model.state_dict(), os.path.join(opt.path_save_model, str(datetime.timestamp(dt)) + 'model' + str(ii.item())))

    print('Final Test Results:')
    print(evaluate(model, test_dataloader, 'test', gpu_id=opt.gpu_id))
    # plot
    plot(epochs, train_mean_losses, ylabel='Loss', name='training-loss-{}-{}'.format(opt.learning_rate, opt.optimizer))
    plot(epochs, valid_specificity, ylabel='Specificity',
         name='validation-specificity-{}-{}'.format(opt.learning_rate, opt.optimizer))
    plot(epochs, valid_sensitivity, ylabel='Sensitivity',
         name='validation-sensitivity-{}-{}'.format(opt.learning_rate, opt.optimizer))


if __name__ == '__main__':
    main()
