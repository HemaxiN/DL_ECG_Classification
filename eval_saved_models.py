import torch
import numpy as np

from torch.utils.data import DataLoader
from utils import Dataset_for_RNN

from lstm import LSTM
from cnn_lstm import CNN1d_LSTM, evaluate
from gru import RNN

gpu_id = None
device = 'cpu'
path_to_data = None  # Dataset/data

# LSTM
# define the model (according to the trained model which weights will be loaded)
model_lstm = LSTM(input_size=3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id,
                  bidirectional=False)
# load the weights
path_to_weights = 'best_trained_rnns/lstm_2lay_128hu'
model_lstm.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))

# 1D-CNN + LSTM (same steps as in LSTM)
model_cnn_lstm = CNN1d_LSTM(input_size=3, hidden_size=128, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id)
path_to_weights = 'best_trained_rnns/cnn_lstm_128hu'
model_cnn_lstm.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))

# GRU (same steps as in LSTM)
model_gru = RNN(3, hidden_size=128, num_layers=3, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id,
                bidirectional=False)
path_to_weights = 'best_trained_rnns/gru_3lay_128hu'
model_gru.load_state_dict(torch.load(path_to_weights, map_location=torch.device(device)))

# choose the model for evaluation on test set
model = model_gru

# model in the evaluation mode
model.eval()

# test dataset
test_dataset = Dataset_for_RNN(path_to_data, [17111, 2156, 2163], 'test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# evaluate the performance of the model
matrix = evaluate(model, test_dataloader, 'test', gpu_id=None)
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
