import torch
import numpy as np

from torch.utils.data import DataLoader
from utils import Dataset_for_RNN

from lstm import LSTM
from cnn_lstm import CNN1d_LSTM, evaluate
from gru import RNN, threshold_optimization, auroc

gpu_id = None
device = 'cpu'
path_to_data = 'data_for_rnn/'  # Dataset/data

# LSTM
# define the model (according to the trained model which weights will be loaded)
# model_lstm = LSTM(input_size=3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id,
#                   bidirectional=False)
# # load the weights
# path_to_weights = 'best_trained_rnns/lstm_2lay_128hu'
# model_lstm.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))
#
# # 1D-CNN + LSTM (same steps as in LSTM)
# model_cnn_lstm = CNN1d_LSTM(input_size=3, hidden_size=128, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id)
# path_to_weights = 'best_trained_rnns/cnn_lstm_128hu'
# model_cnn_lstm.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))

# GRU (same steps as in LSTM)
model_bigru = RNN(3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0.5, gpu_id=gpu_id,
                bidirectional=True)
path_to_weights = 'best_trained_rnns/grubi_dropout05_lr0005_model5'
model_bigru.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))

# choose the model for evaluation on test set
model = model_bigru

# model in the evaluation mode
model.eval()

# test dataset
test_dataset = Dataset_for_RNN(path_to_data, [17111, 2156, 2163], 'test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# dev dataset
dev_dataset = Dataset_for_RNN(path_to_data, [17111, 2156, 2163], 'dev')
dev_dataloader = DataLoader(dev_dataset, batch_size=1024, shuffle=False)

# threshold optimization
thr = threshold_optimization(model, dev_dataloader)

# evaluate the performance of the model
matrix = evaluate(model, test_dataloader, thr, gpu_id=None)
aurocs = auroc(model, test_dataloader, gpu_id=gpu_id)
MI_sensi = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
MI_spec = matrix[0, 3] / (matrix[0, 3] + matrix[0, 2])
STTC_sensi = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
STTC_spec = matrix[1, 3] / (matrix[1, 3] + matrix[1, 2])
CD_sensi = matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])
CD_spec = matrix[2, 3] / (matrix[2, 3] + matrix[2, 2])
HYP_sensi = matrix[3, 0] / (matrix[3, 0] + matrix[3, 1])
HYP_spec = matrix[3, 3] / (matrix[3, 3] + matrix[3, 2])
mean_sensi = np.mean(matrix[:, 0]) / (np.mean(matrix[:, 0]) + np.mean(matrix[:, 1]))
mean_spec = np.mean(matrix[:, 3]) / (np.mean(matrix[:, 3]) + np.mean(matrix[:, 2]))
print('Final Test Results: \n ' + str(matrix) + '\n' + 'MI: sensitivity - ' + str(MI_sensi) + '; specificity - '
      + str(MI_spec) + '\n' + 'STTC: sensitivity - ' + str(STTC_sensi) + '; specificity - ' + str(STTC_spec)
      + '\n' + 'CD: sensitivity - ' + str(CD_sensi) + '; specificity - ' + str(CD_spec)
      + '\n' + 'HYP: sensitivity - ' + str(HYP_sensi) + '; specificity - ' + str(HYP_spec)
      + '\n' + 'mean: sensitivity - ' + str(mean_sensi) + '; specificity - ' + str(mean_spec))


MI_sensi = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
MI_spec = matrix[0, 3] / (matrix[0, 3] + matrix[0, 2])
MI_acc = (matrix[0, 0] + matrix[0, 3]) / np.sum(matrix[0])
MI_prec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 2])
MI_f1 = (2 * matrix[0, 0]) / (2 * matrix[0, 0] + matrix[0, 2] + matrix[0, 1])
MI_auroc = aurocs[0].item()


STTC_sensi = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
STTC_spec = matrix[1, 3] / (matrix[1, 3] + matrix[1, 2])
STTC_acc = (matrix[1, 0] + matrix[1, 3]) / np.sum(matrix[1])
STTC_prec = matrix[1, 0] / (matrix[1, 0] + matrix[1, 2])
STTC_f1 = (2 * matrix[1, 0]) / (2 * matrix[1, 0] + matrix[1, 2] + matrix[1, 1])
STTC_auroc = aurocs[1].item()

CD_sensi = matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])
CD_spec = matrix[2, 3] / (matrix[2, 3] + matrix[2, 2])
CD_acc = (matrix[2, 0] + matrix[2, 3]) / np.sum(matrix[2])
CD_prec = matrix[2, 0] / (matrix[2, 0] + matrix[2, 2])
CD_f1 = (2 * matrix[2, 0]) / (2 * matrix[2, 0] + matrix[2, 2] + matrix[2, 1])
CD_auroc = aurocs[2].item()

HYP_sensi = matrix[3, 0] / (matrix[3, 0] + matrix[3, 1])
HYP_spec = matrix[3, 3] / (matrix[3, 3] + matrix[3, 2])
HYP_acc = (matrix[3, 0] + matrix[3, 3]) / np.sum(matrix[3])
HYP_prec = matrix[3, 0] / (matrix[3, 0] + matrix[3, 2])
HYP_f1 = (2 * matrix[3, 0]) / (2 * matrix[3, 0] + matrix[3, 2] + matrix[3, 1])
HYP_auroc = aurocs[3].item()

# compute mean sensitivity and specificity:
mean_mat = np.mean(matrix, axis=0)
mean_sensi = mean_mat[0] / (mean_mat[0] + mean_mat[1])
mean_spec = mean_mat[3] / (mean_mat[3] + mean_mat[2])
mean_acc = (mean_mat[0] + mean_mat[3]) / np.sum(mean_mat)
mean_prec = mean_mat[0] / (mean_mat[0] + mean_mat[2])
mean_f1 = (2 * mean_mat[0]) / (2 * mean_mat[0] + mean_mat[2] + mean_mat[1])
mean_auroc = aurocs.mean().item()
mean_g = np.sqrt(mean_spec * mean_sensi)

print('Final Test Results: \n ' + str(matrix) + '\n\n' +
      'MI: \n\tsensitivity - ' + str(MI_sensi) + '\n\tspecificity - ' + str(MI_spec) + '\n\tprecision - ' + str(MI_prec) + '\n\taccuracy - ' + str(MI_acc) + '\n\tF1 Score - ' + str(MI_f1) + '\n\tAUROC - ' + str(MI_auroc) + '\n' +
      'STTC: \n\tsensitivity - ' + str(STTC_sensi) + '\n\tspecificity - ' + str(STTC_spec) + '\n\tprecision - ' + str(STTC_prec) + '\n\taccuracy - ' + str(STTC_acc) + '\n\tF1 Score - ' + str(STTC_f1) + '\n\tAUROC - ' + str(STTC_auroc) + '\n' +
      'CD: \n\tsensitivity - ' + str(CD_sensi) + '\n\tspecificity - ' + str(CD_spec) + '\n\tprecision - ' + str(CD_prec) + '\n\taccuracy - ' + str(CD_acc) + '\n\tF1 Score - ' + str(CD_f1) + '\n\tAUROC - ' + str(CD_auroc) + '\n' +
      'HYP: \n\tsensitivity - ' + str(HYP_sensi) + '\n\tspecificity - ' + str(HYP_spec) + '\n\tprecision - ' + str(HYP_prec) + '\n\taccuracy - ' + str(HYP_acc) + '\n\tF1 Score - ' + str(HYP_f1) + '\n\tAUROC - ' + str(HYP_auroc) + '\n' +
      'mean: \n\tG-Mean - ' + str(mean_g) + '\n\tsensitivity - ' + str(mean_sensi) + '\n\tspecificity - ' + str(mean_spec) + '\n\tprecision - ' + str(mean_prec) + '\n\taccuracy - ' + str(mean_acc) + '\n\tF1 Score - ' + str(mean_f1) + '\n\tAUROC - ' + str(mean_auroc))

print('Final Test Results: \n ' + str(matrix) + '\n\n' +
      'MI: \n\tsensitivity - ' + str(MI_sensi) + '\n\tspecificity - ' + str(MI_spec) + '\n\tprecision - ' + str(MI_prec) + '\n\taccuracy - ' + str(MI_acc) + '\n\tF1 Score - ' + str(MI_f1) + '\n\tAUROC - ' + str(MI_auroc) + '\n' +
      'STTC: \n\tsensitivity - ' + str(STTC_sensi) + '\n\tspecificity - ' + str(STTC_spec) + '\n\tprecision - ' + str(STTC_prec) + '\n\taccuracy - ' + str(STTC_acc) + '\n\tF1 Score - ' + str(STTC_f1) + '\n\tAUROC - ' + str(STTC_auroc) + '\n' +
      'CD: \n\tsensitivity - ' + str(CD_sensi) + '\n\tspecificity - ' + str(CD_spec) + '\n\tprecision - ' + str(CD_prec) + '\n\taccuracy - ' + str(CD_acc) + '\n\tF1 Score - ' + str(CD_f1) + '\n\tAUROC - ' + str(CD_auroc) + '\n' +
      'HYP: \n\tsensitivity - ' + str(HYP_sensi) + '\n\tspecificity - ' + str(HYP_spec) + '\n\tprecision - ' + str(HYP_prec) + '\n\taccuracy - ' + str(HYP_acc) + '\n\tF1 Score - ' + str(HYP_f1) + '\n\tAUROC - ' + str(HYP_auroc) + '\n' +
      'mean: \n\tG-Mean - ' + str(mean_g) + '\n\tsensitivity - ' + str(mean_sensi) + '\n\tspecificity - ' + str(mean_spec) + '\n\tprecision - ' + str(mean_prec) + '\n\taccuracy - ' + str(mean_acc) + '\n\tF1 Score - ' + str(mean_f1) + '\n\tAUROC - ' + str(mean_auroc))
