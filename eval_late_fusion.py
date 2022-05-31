import torch
import numpy as np

from torch.utils.data import DataLoader
import gru as gru
import AlexNet as alexnet
import late_fusion as late
import early_fusion as early
import joint_fusion as joint

type = ['late', 'early', 'joint'][2]

batch_size = 64
hidden_size = 256
dropout = 0.3
# path_weights = "save_models/1654007792.211898late_model9"
# path_weights = "save_models/1654009322early_model2"
path_weights = "save_models/1654008252.033359joint_256hl_model1"

gpu_id = 0

sig_data = "Dataset/data_for_rnn/"
sig_path = "best_trained_rnns/gru_3lay_128hu"
sig_model = gru.RNN(3, 128, 3, 4, .3, gpu_id=gpu_id, bidirectional=False).to(gpu_id)

img_data = "Dataset/Images/"
img_path = "Models/alexnet"
img_model = alexnet.AlexNet(4).to(gpu_id)

sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))

if type == 'late':
    sig_model.eval()
    img_model.eval()

    test_dataset = late.LateFusionDataset(sig_data, img_data, sig_model, img_model, 'gru', 'alexnet',
                                          [17111, 2156, 2163], gpu_id, batch_size, part='test')

    model = late.LateFusionNet(4, 8, hidden_size, dropout).to(gpu_id)

elif type == 'early':
    sig_model.eval()
    img_model.eval()

    img_hook = 'conv2d_5'
    sig_hook = 'rnn'
    img_model.conv2d_5.register_forward_hook(early.get_activation(img_hook))
    sig_model.rnn.register_forward_hook(early.get_activation(sig_hook))

    test_dataset = early.FusionDataset(sig_data, img_data, [17111, 2156, 2163], part='test')

    model = early.EarlyFusionNet(4, 128, 4096, hidden_size, dropout,
                                 sig_model, img_model, sig_hook, img_hook).to(gpu_id)

else:  # joint fusion
    sig_model.fc = joint.Identity()
    img_model.linear_3 = joint.Identity()

    test_dataset = early.FusionDataset(sig_data, img_data, [17111, 2156, 2163], part='test')

    model = joint.JointFusionNet(4, 128, 2048, hidden_size, dropout, sig_model, img_model).to(gpu_id)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.load_state_dict(torch.load(path_weights, map_location=torch.device(gpu_id)))
model = model.to(gpu_id)
model.eval()

# evaluate the performance of the model
if type == 'late':
    matrix = gru.evaluate(model, test_dataloader, 'test', gpu_id=gpu_id)
else:
    matrix = early.fusion_evaluate(model, test_dataloader, 'test', gpu_id=gpu_id)

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

