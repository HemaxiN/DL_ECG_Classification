import torch
import numpy as np

from torch.utils.data import DataLoader
import gru as gru
import AlexNet as alexnet
import late_fusion as late
import early_fusion as early
import joint_fusion as joint

from count_parameters import count_parameters

type = ['late', 'early', 'joint'][0]

if type == "joint":
    path_weights = "save_models/joint_model_2023-02-03_06-54-57_lr0.01_optadam_dr0.3_eps200_hs256_bs1024_l20.0001"
    thresholds = [0.4761, 0.656, 0.7849, 0.8079]
    batch_size = 1024
    hidden_size = 256
    dropout = 0

elif type == "early":
    path_weights = "save_models/early_model_2023-01-26_01-04-33_lr0.001_optadam_dr0.0_eps200_hs256_bs128_l20"
    thresholds = [0.5659, 0.3952, 0.6742, 0.769]

    batch_size = 128
    hidden_size = 256
    dropout = 0

elif type == "late":
    path_weights = "save_models/late_model_2023-01-22_04-30-10_lr0.1_optadam_dr0.3_eps200_hs512_bs512_l20"
    thresholds = [0.3931, 0.645, 0.6972, 0.8535]

    batch_size = 512
    hidden_size = 512
    dropout = 0

sig_path = "save_models/grubi_dropout05_lr0005_model5"
img_path = "save_models/alexnet"

gpu_id = 0

sig_data = "Dataset/data_for_rnn/"
sig_model = gru.RNN(3, 128, 2, 4, 0.5, gpu_id=gpu_id,
                    bidirectional=True).to(gpu_id)

img_data = "Dataset/Images/"
img_model = alexnet.AlexNet(4).to(gpu_id)

if type == 'late':
    sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
    img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
    sig_model.eval()
    img_model.eval()

    test_dataset = late.LateFusionDataset(sig_data, img_data, sig_model, img_model, 'gru', 'alexnet',
                                          [17111, 2156, 2163], gpu_id, batch_size, part='test')

    model = late.LateFusionNet(4, 8, hidden_size, dropout).to(gpu_id)

elif type == 'early':
    sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
    sig_model.requires_grad_(False)
    img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
    img_model.requires_grad_(False)
    sig_model.eval()
    img_model.eval()

    img_hook = 'conv2d_5'
    sig_hook = 'rnn'
    img_model.conv2d_5.register_forward_hook(early.get_activation(img_hook))
    sig_model.rnn.register_forward_hook(early.get_activation(sig_hook))

    test_dataset = early.FusionDataset(sig_data, img_data, [17111, 2156, 2163], part='test')

    model = early.EarlyFusionNet(4, 256, 4096, hidden_size, dropout,
                                 sig_model, img_model, sig_hook, img_hook).to(gpu_id)

else:  # joint fusion
    sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
    img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
    sig_model.fc = joint.Identity()
    img_model.linear_3 = joint.Identity()

    test_dataset = early.FusionDataset(sig_data, img_data, [17111, 2156, 2163], part='test')

    model = joint.JointFusionNet(4, 256, 2048, hidden_size, dropout, sig_model, img_model).to(gpu_id)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.load_state_dict(torch.load(path_weights, map_location=torch.device(gpu_id)))
model = model.to(gpu_id)
model.eval()

# evaluate the performance of the model
if type == 'late':
    matrix = gru.evaluate(model, test_dataloader, thresholds, gpu_id=gpu_id)
    aurocs = gru.auroc(model, test_dataloader, gpu_id=gpu_id)
else:
    matrix = early.fusion_evaluate(model, test_dataloader, thresholds, gpu_id=gpu_id)
    aurocs = early.fusion_auroc(model, test_dataloader, gpu_id=gpu_id)

count_parameters(model)

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

