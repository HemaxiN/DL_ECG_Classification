# Code based on the source code of homework 1 and homework 2 of the
# deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import configure_seed, configure_device, compute_scores, Dataset_for_RNN, \
    plot_losses
from datetime import datetime
import statistics
import numpy as np
import os
from sklearn.metrics import roc_curve
from torchmetrics.classification import MultilabelAUROC


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Define the attention layer
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output):
        # rnn_output shape: (batch_size, seq_length, hidden_size) if batch_first=True
        # rnn_output shape: (seq_length, batch_size, hidden_size) if batch_first=False
        if not self.batch_first:
            rnn_output = rnn_output.transpose(0, 1)  # (batch_size, seq_length, hidden_size)

        # Apply attention layer to the hidden states
        attn_weights = self.attention(rnn_output)  # (batch_size, seq_length, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Multiply the weights by the rnn_output to get a weighted sum
        context = torch.sum(attn_weights * rnn_output, dim=1)  # (batch_size, hidden_size)
        return context, attn_weights


class RNN_att(nn.Module):
    # ... [previous __init__ definition] ...

    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout_rate, bidirectional, gpu_id=None):
        """
        Define the layers of the model
        Args:
            input_size (int): "Feature" size (in this case, it is 3)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of hidden RNN layers
            n_classes (int): Number of classes in our classification problem
            dropout_rate (float): Dropout rate to be applied in all rnn layers except the last one
            bidirectional (bool): Boolean value: if true, gru layers are bidirectional
        """
        super(RNN_att, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.gpu_id = gpu_id
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True,
                          bidirectional=bidirectional)  # batch_first: first dimension is the batch size

        if bidirectional:
            self.d = 2
        else:
            self.d = 1

        # Initialize the attention layer
        self.attention = Attention(hidden_size * self.d, batch_first=True)

        # Adjust the input dimension for the classification layer according to bidirectionality
        self.fc = nn.Linear(hidden_size * self.d, n_classes)

    def forward(self, X):
        """
                Forward Propagation

                Args:
                    X: batch of training examples with dimension (batch_size, 1000, 3)
                """
        # initial hidden state:
        h_0 = torch.zeros(self.num_layers * self.d, X.size(0), self.hidden_size).to(self.gpu_id)

        out_rnn, _ = self.rnn(X.to(self.gpu_id), h_0)
        # out_rnn shape: (batch_size, seq_length, hidden_size*d) = (batch_size, 1000, hidden_size*d)

        # when we apply the attention mechanism, we want to apply it to the whole sequence, so the next part is deleted
        # if self.bidirectional:
        #     # concatenate last timestep from the "left-to-right" direction and the first timestep from the
        #     # "right-to-left" direction
        #     out_rnn = torch.cat((out_rnn[:, -1, :self.hidden_size], out_rnn[:, 0, self.hidden_size:]), dim=1)
        # else:
        #     # last timestep
        #     out_rnn = out_rnn[:, -1, :]

        # Apply attention
        attn_output, attn_weights = self.attention(out_rnn)
        # attn_output shape: (batch_size, hidden_size*d)

        # Output layer
        out_fc = self.fc(attn_output)
        # out_fc shape: (batch_size, n_classes)

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


def predict(model, X, thr):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    logits_ = model(X)  # (batch_size, n_classes)
    probabilities = torch.sigmoid(logits_).cpu()

    if thr is None:
        return probabilities
    else:
        return np.array(probabilities.numpy() >= thr, dtype=float)


def evaluate(model, dataloader, thr, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,4): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        matrix = np.zeros((4, 4))
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = predict(model, x_batch, thr)
            y_true = np.array(y_batch.cpu())
            matrix = compute_scores(y_true, y_pred, matrix)

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

    return matrix
    # cols: TP, FN, FP, TN


def auroc(model, dataloader, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,4): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        preds = []
        trues = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)

            preds += predict(model, x_batch, None)
            trues += [y_batch.cpu()[0]]

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

    preds = torch.stack(preds)
    trues = torch.stack(trues).int()
    return MultilabelAUROC(num_labels=4, average=None)(preds, trues)
    # cols: TP, FN, FP, TN


# Validation loss
def compute_loss(model, dataloader, criterion, gpu_id=None):
    model.eval()
    with torch.no_grad():
        val_losses = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_losses.append(loss.item())
            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

        return statistics.mean(val_losses)


def threshold_optimization(model, dataloader, gpu_id=None):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    save_probs = []
    save_y = []
    threshold_opt = np.zeros(4)

    model.eval()
    with torch.no_grad():
        #threshold_opt = np.zeros(4)
        for _, (X, Y) in enumerate(dataloader):
            X, Y = X.to(gpu_id), Y.to(gpu_id)

            Y = np.array(Y.cpu())
            #print(Y)

            logits_ = model(X)  # (batch_size, n_classes)
            probabilities = torch.sigmoid(logits_).cpu()

            save_probs += [probabilities.numpy()]
            save_y += [Y]

    # find the optimal threshold with ROC curve for each disease

    save_probs = np.array(np.concatenate(save_probs)).reshape((-1, 4))
    save_y = np.array(np.concatenate(save_y)).reshape((-1, 4))
    for dis in range(0, 4):
        # print(probabilities[:, dis])
        # print(Y[:, dis])
        fpr, tpr, thresholds = roc_curve(save_y[:, dis], save_probs[:, dis])
        # geometric mean of sensitivity and specificity
        gmean = np.sqrt(tpr * (1 - fpr))
        # optimal threshold
        index = np.argmax(gmean)
        threshold_opt[dis] = round(thresholds[index], ndigits=2)

    return threshold_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data_for_rnn/',
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=200, type=int,
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
    parser.add_argument('-hidden_size', type=int, default=128)
    parser.add_argument('-bidirectional', type=bool, default=False)
    parser.add_argument('-early_stop', type=bool, default=True)
    parser.add_argument('-patience', type=int, default=20)
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    samples = [17111, 2156, 2163]
    print("Loading data...")
    train_dataset = Dataset_for_RNN(opt.data, samples, 'train')
    dev_dataset = Dataset_for_RNN(opt.data, samples, 'dev')
    test_dataset = Dataset_for_RNN(opt.data, samples, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    dev_dataloader_thr = DataLoader(dev_dataset, batch_size=1024, shuffle=False)

    input_size = 3
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    n_classes = 4

    # initialize the model
    model = RNN_att(input_size, hidden_size, num_layers, n_classes, dropout_rate=opt.dropout, gpu_id=opt.gpu_id,
                bidirectional=opt.bidirectional)
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
    class_weights = torch.tensor([17111/4389, 17111/3136, 17111/1915, 17111/417], dtype=torch.float)
    class_weights = class_weights.to(opt.gpu_id)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights)  # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_mean_losses = []
    train_losses = []
    epochs_run = opt.epochs
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion, gpu_id=opt.gpu_id)
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        val_loss = compute_loss(model, dev_dataloader, criterion, gpu_id=opt.gpu_id)
        valid_mean_losses.append(val_loss)
        print('Validation loss: %.4f' % (val_loss))

        dt = datetime.now()
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save the model at each epoch where the validation loss is the best so far
        if val_loss == np.min(valid_mean_losses):
            f = os.path.join(opt.path_save_model, str(datetime.timestamp(dt)) + 'model' + str(ii.item()))
            best_model = ii
            torch.save(model.state_dict(), f)

        # early stop - if validation loss does not increase for 15 epochs, stop learning process
        if opt.early_stop:
            if ii > opt.patience:
                if valid_mean_losses[ii - opt.patience] == np.min(valid_mean_losses[ii - opt.patience:]):
                    epochs_run = ii
                    break

    # Make predictions based on best model (lowest validation loss)
    # Load model
    model.load_state_dict(torch.load(f))
    model.eval()

    # Threshold optimization on validation set
    thr = threshold_optimization(model, dev_dataloader_thr)

    # Results on test set:
    matrix = evaluate(model, test_dataloader, thr, gpu_id=opt.gpu_id)

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

    dt = datetime.now()
    with open('results/' + 'model' + str(best_model.item()) + '_' + str(datetime.timestamp(dt)) + '.txt', 'w') as f:
        print('Final Test Results: \n ' + str(matrix) + '\n' + 'MI: sensitivity - ' + str(MI_sensi) + '; specificity - '
              + str(MI_spec) + '\n' + 'STTC: sensitivity - ' + str(STTC_sensi) + '; specificity - ' + str(STTC_spec)
              + '\n' + 'CD: sensitivity - ' + str(CD_sensi) + '; specificity - ' + str(CD_spec)
              + '\n' + 'HYP: sensitivity - ' + str(HYP_sensi) + '; specificity - ' + str(HYP_spec)
              + '\n' + 'mean: sensitivity - ' + str(mean_sensi) + '; specificity - ' + str(mean_spec), file=f)

    # plot
    epochs_axis = torch.arange(1, epochs_run + 1)
    plot_losses(valid_mean_losses, train_mean_losses, ylabel='Loss',
                name='training-validation-loss-{}-{}-{}'.format(opt.learning_rate, opt.optimizer, dt))


if __name__ == '__main__':
    main()
