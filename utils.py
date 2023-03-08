# code based on the source code of homework 1 and homework 2 of the
# deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks

# import the necessary packages
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import csv
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


def plot(plottable, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(plottable)
    plt.savefig("%s.pdf" % (name), bbox_inches="tight")


def plot_losses(valid_losses, train_losses, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    # plt.xticks(epochs)
    plt.plot(valid_losses, label="validation")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.savefig("%s.pdf" % (name), bbox_inches="tight")


# create a generator to read the images as we train the model
# (similar to flow_from_directory Keras)
class ECGImageDataset(Dataset):
    """
    path/train/images
              /labels
        /val/images
            /labels
        /test/images
             /labels
    """

    def __init__(self, path, train_dev_test, part="train"):
        self.path = path
        self.part = part
        self.train_dev_test = train_dev_test

    def __len__(self):
        if self.part == "train":
            return self.train_dev_test[0]
        elif self.part == "dev":
            return self.train_dev_test[1]
        elif self.part == "test":
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        X, y = read_data_for_CNN(self.path, self.part, idx)
        return torch.tensor(X).float(), torch.tensor(y).float()


def read_data_for_CNN(path, partition, idx):
    """Read the ECG Image Data"""
    path_labels = str(path) + "labels_" + str(partition)
    path_X = str(path) + "X_cnn_" + str(partition)
    index = idx
    label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    image = tifffile.imread(str(path_X) + "/" + str(index) + ".tif")
    image = image / 255.0  # normalization
    return image, label


class Dataset_for_RNN(Dataset):
    """
    path/labels_train
        /X_train
        /labels_val
        /X_val
        /labels_test
        /X_test
    """

    def __init__(self, path, train_dev_test, part="train"):
        self.path = path
        self.part = part
        self.train_dev_test = train_dev_test

    def __len__(self):
        if self.part == "train":
            return self.train_dev_test[0]
        elif self.part == "dev":
            return self.train_dev_test[1]
        elif self.part == "test":
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        X, y = read_data_for_RNN(self.path, self.part, idx)
        return torch.tensor(X).float(), torch.tensor(y).float()


def read_data_for_RNN(path, partition, idx):
    path_labels = str(path) + "labels_" + str(partition)
    path_X = str(path) + "X_rnn_" + str(partition)
    index = idx
    label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    X = np.load(str(path_X) + "/" + str(index) + ".npy")
    return X, label


# performance evaluation, compute the tp, fn, fp, and tp for each disease class
# and compute the specificity and sensitivity
def compute_scores(y_true, y_pred, matrix):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        for i in range(0, 4):  # for each class
            matrix = computetpfnfp(pred[i], gt[i], i, matrix)
    return matrix


def compute_scores_dev(matrix):
    matrix[matrix == 0] = 0.01
    # print(matrix)
    sensitivity = matrix[:, 0] / (matrix[:, 0] + matrix[:, 1])  # tp/(tp+fn)
    specificity = matrix[:, 3] / (matrix[:, 3] + matrix[:, 2])  # tn/(tn+fp)
    return np.mean(sensitivity), np.mean(specificity)


def computetpfnfp(pred, gt, i, matrix):
    if gt == 0 and pred == 0:  # tn
        matrix[i, 3] += 1
    if gt == 1 and pred == 0:  # fn
        matrix[i, 1] += 1
    if gt == 0 and pred == 1:  # fp
        matrix[i, 2] += 1
    if gt == 1 and pred == 1:  # tp
        matrix[i, 0] += 1
    return matrix


def compute_save_metrics(matrix, matrix_dev, opt_threshold, date, epoch, strategy, path_save_model, learning_rate,
                         optimizer, dropout, epochs, hidden_size, batch_size, test_id):

    # compute sensitivity and specificity for each class:
    MI_sensi = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    MI_spec = matrix[0, 3] / (matrix[0, 3] + matrix[0, 2])
    STTC_sensi = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
    STTC_spec = matrix[1, 3] / (matrix[1, 3] + matrix[1, 2])
    CD_sensi = matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])
    CD_spec = matrix[2, 3] / (matrix[2, 3] + matrix[2, 2])
    HYP_sensi = matrix[3, 0] / (matrix[3, 0] + matrix[3, 1])
    HYP_spec = matrix[3, 3] / (matrix[3, 3] + matrix[3, 2])

    MI_sensi_dev = matrix_dev[0, 0] / (matrix_dev[0, 0] + matrix_dev[0, 1])
    MI_spec_dev = matrix_dev[0, 3] / (matrix_dev[0, 3] + matrix_dev[0, 2])
    STTC_sensi_dev = matrix_dev[1, 0] / (matrix_dev[1, 0] + matrix_dev[1, 1])
    STTC_spec_dev = matrix_dev[1, 3] / (matrix_dev[1, 3] + matrix_dev[1, 2])
    CD_sensi_dev = matrix_dev[2, 0] / (matrix_dev[2, 0] + matrix_dev[2, 1])
    CD_spec_dev = matrix_dev[2, 3] / (matrix_dev[2, 3] + matrix_dev[2, 2])
    HYP_sensi_dev = matrix_dev[3, 0] / (matrix_dev[3, 0] + matrix_dev[3, 1])
    HYP_spec_dev = matrix_dev[3, 3] / (matrix_dev[3, 3] + matrix_dev[3, 2])

    # compute mean sensitivity and specificity:
    mean_sensi = np.mean(matrix[:, 0]) / (np.mean(matrix[:, 0]) + np.mean(matrix[:, 1]))
    mean_spec = np.mean(matrix[:, 3]) / (np.mean(matrix[:, 3]) + np.mean(matrix[:, 2]))
    mean_sensi_dev = np.mean(matrix_dev[:, 0]) / (np.mean(matrix_dev[:, 0]) + np.mean(matrix_dev[:, 1]))
    mean_spec_dev = np.mean(matrix_dev[:, 3]) / (np.mean(matrix_dev[:, 3]) + np.mean(matrix_dev[:, 2]))

    # print results:
    print(
        "Final Validation Results: \n "
        + str(matrix_dev)
        + "\n"
        + "MI: sensitivity - "
        + str(MI_sensi_dev)
        + "; specificity - "
        + str(MI_spec_dev)
        + "\n"
        + "STTC: sensitivity - "
        + str(STTC_sensi_dev)
        + "; specificity - "
        + str(STTC_spec_dev)
        + "\n"
        + "CD: sensitivity - "
        + str(CD_sensi_dev)
        + "; specificity - "
        + str(CD_spec_dev)
        + "\n"
        + "HYP: sensitivity - "
        + str(HYP_sensi_dev)
        + "; specificity - "
        + str(HYP_spec_dev)
        + "\n"
        + "mean: sensitivity - "
        + str(mean_sensi_dev)
        + "; specificity - "
        + str(mean_spec_dev)
    )

    print(
        "Final Test Results: \n "
        + str(matrix)
        + "\n"
        + "MI: sensitivity - "
        + str(MI_sensi)
        + "; specificity - "
        + str(MI_spec)
        + "\n"
        + "STTC: sensitivity - "
        + str(STTC_sensi)
        + "; specificity - "
        + str(STTC_spec)
        + "\n"
        + "CD: sensitivity - "
        + str(CD_sensi)
        + "; specificity - "
        + str(CD_spec)
        + "\n"
        + "HYP: sensitivity - "
        + str(HYP_sensi)
        + "; specificity - "
        + str(HYP_spec)
        + "\n"
        + "mean: sensitivity - "
        + str(mean_sensi)
        + "; specificity - "
        + str(mean_spec)
    )

    with open(
        "{}{}{}_{}_ep{}_lr{}_opt{}_dr{}_eps{}_hs{}_bs{}.txt".format(
            path_save_model,
            test_id,
            strategy,
            date,
            epoch.item(),
            learning_rate,
            optimizer,
            dropout,
            epochs,
            hidden_size,
            batch_size,
        ),
        "w",
    ) as f:
        f.write("Final Results\n\n")
        f.write("Threshold: {}\n\n".format(np.round(opt_threshold, 4)))

        f.write("Development/Validation\n")
        f.write("MI\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(MI_sensi_dev, MI_spec_dev))
        f.write("STTC\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(STTC_sensi_dev, STTC_spec_dev))
        f.write("CD\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(CD_sensi_dev, CD_spec_dev))
        f.write("HYP\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(HYP_sensi_dev, HYP_spec_dev))
        f.write("Mean\n\tSensitivity: {}\n\tSpecificity: {}\n\n\n".format(mean_sensi_dev, mean_spec_dev))

        f.write("Test\n")
        f.write("MI\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(MI_sensi, MI_spec))
        f.write("STTC\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(STTC_sensi, STTC_spec))
        f.write("CD\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(CD_sensi, CD_spec))
        f.write("HYP\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(HYP_sensi, HYP_spec))
        f.write("Mean\n\tSensitivity: {}\n\tSpecificity: {}".format(mean_sensi, mean_spec))

    fields = [test_id,
              strategy,
              date,
              epoch.item(),
              learning_rate,
              optimizer,
              dropout,
              epochs,
              hidden_size,
              batch_size,
              mean_sensi_dev,
              mean_spec_dev,
              mean_sensi,
              mean_spec
              ]

    with open(path_save_model + "auto_results.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
