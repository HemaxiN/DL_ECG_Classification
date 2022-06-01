import torch
import numpy as np
from prettytable import PrettyTable

from lstm import LSTM
from cnn_lstm import CNN1d_LSTM
from gru import RNN


# create a table with the model's parameters
# code from the comments https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


gpu_id = None
device = 'cpu'

# LSTM
# model_lstm = LSTM(input_size=3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id,
#                   bidirectional=False)
# model_lstm.load_state_dict(torch.load('best_trained_rnns/lstm_2lay_128hu', map_location=torch.device(device)))
#
# # 1D-CNN + LSTM
# model_cnn_lstm = CNN1d_LSTM(input_size=3, hidden_size=128, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id)
# model_cnn_lstm.load_state_dict(torch.load('best_trained_rnns/cnn_lstm_128hu', map_location=torch.device(device)))
#
# # GRU
# model_gru = RNN(3, hidden_size=128, num_layers=3, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id,
#                 bidirectional=False)
# model_gru.load_state_dict(torch.load('best_trained_rnns/gru_3lay_128hu', map_location=torch.device(device)))
#
# count_parameters(model_lstm)
# count_parameters(model_gru)
# count_parameters(model_cnn_lstm)

