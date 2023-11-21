import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import Dataset_for_RNN, configure_device, configure_seed
import argparse
from lstm import LSTM
from cnn_lstm import CNN1d_LSTM
from cnn_gru import CNN1d_GRU
from gru_with_attention import RNN_att
from gru import RNN, threshold_optimization, auroc, evaluate_with_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-model', type=str, default='gru')
    opt = parser.parse_args()

    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    gpu_id = opt.gpu_id
    device = 'cuda'
    if gpu_id is None:
        device = 'cpu'

    path_to_data = 'data_for_rnn/'  # Dataset/data

    # choose the model for evaluation on test set

    if opt.model == 'lstm':
        # LSTM
        model_lstm = LSTM(input_size=3, hidden_size=256, num_layers=2, n_classes=4, dropout_rate=0, gpu_id=gpu_id,
                          bidirectional=False)
        model_lstm.load_state_dict(
            torch.load('best_trained_rnns/lstm_1669240561.183005model28'))
        model = model_lstm.to(opt.gpu_id)
    elif opt.model == 'gru':
        # GRU
        model_gru = RNN(3, hidden_size=128, num_layers=3, n_classes=4, dropout_rate=0, gpu_id=gpu_id,
                        bidirectional=False)
        model_gru.load_state_dict(
            torch.load('best_trained_rnns/gru_3layers_dropout0_model8'))
        model = model_gru.to(opt.gpu_id)
    elif opt.model == 'bigru':
        # BiGRU
        model_bigru = RNN(3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0.5, gpu_id=gpu_id,
                          bidirectional=True)
        model_bigru.load_state_dict(
            torch.load('best_trained_rnns/grubi_dropout05_lr0005_model5'))
        model = model_bigru.to(opt.gpu_id)
    elif opt.model == 'bigruattention':
        # BiGRU
        model_bigru_att = RNN_att(3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0.5, gpu_id=gpu_id,
                          bidirectional=True)
        model_bigru_att.load_state_dict(
            torch.load('best_trained_rnns/grubi_attention_model4'))
        model = model_bigru_att.to(opt.gpu_id)
    elif opt.model == 'gruattention':
        # BiGRU
        model_gru_att = RNN_att(3, hidden_size=128, num_layers=3, n_classes=4, dropout_rate=0, gpu_id=gpu_id,
                                  bidirectional=False)
        model_gru_att.load_state_dict(
            torch.load('best_trained_rnns/gru_attention_model4'))
        model = model_gru_att.to(opt.gpu_id)
    elif opt.model == 'cnn_gru':
        # 1D-CNN + GRU
        model_cnn_gru = CNN1d_GRU(input_size=3, hidden_size=256, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id)
        model_cnn_gru.load_state_dict(
            torch.load('best_trained_rnns/cnn_gru_1672230038.517979model49'))
        model = model_cnn_gru.to(opt.gpu_id)
    elif opt.model == 'cnn_lstm':
        # 1D-CNN + LSTM
        model_cnn_lstm = CNN1d_LSTM(input_size=3, hidden_size=128, n_classes=4, dropout_rate=0.3, gpu_id=gpu_id)
        model_cnn_lstm.load_state_dict(
            torch.load('best_trained_rnns/cnnlstm_model120'))
        model = model_cnn_lstm.to(opt.gpu_id)
    elif opt.model == 'bilstm':
        # BiLSTM
        model_bilstm = LSTM(input_size=3, hidden_size=128, num_layers=2, n_classes=4, dropout_rate=0, gpu_id=gpu_id,
                            bidirectional=True)
        model_bilstm.load_state_dict(
            torch.load('best_trained_rnns/lstmbi_dropout05_model20'))
        model = model_bilstm.to(opt.gpu_id)

    # model in the evaluation mode
    model.eval()

    # test dataset
    test_dataset = Dataset_for_RNN(path_to_data, [17111, 2156, 2163], 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # dev dataset
    dev_dataset = Dataset_for_RNN(path_to_data, [17111, 2156, 2163], 'dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=512, shuffle=False)

    # threshold optimization
    thr = threshold_optimization(model, dev_dataloader)

    # evaluate the performance of the model
    matrix, norm_vec = evaluate_with_norm(model, test_dataloader, thr, gpu_id=gpu_id)
    # matrix = evaluate(model, test_dataloader, thr, gpu_id=None)
    aurocs = auroc(model, test_dataloader, gpu_id=gpu_id)

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

    NORM_sensi = norm_vec[0] / (norm_vec[0] + norm_vec[1])
    NORM_spec = norm_vec[3] / (norm_vec[3] + norm_vec[2])
    NORM_acc = (norm_vec[0] + norm_vec[3]) / np.sum(matrix[3])
    NORM_prec = norm_vec[0] / (norm_vec[0] + norm_vec[2])
    NORM_f1 = (2 * norm_vec[0]) / (2 * norm_vec[0] + norm_vec[2] + norm_vec[1])

    # compute mean sensitivity and specificity:
    mean_mat = np.mean(matrix, axis=0)
    mean_sensi = mean_mat[0] / (mean_mat[0] + mean_mat[1])
    mean_spec = mean_mat[3] / (mean_mat[3] + mean_mat[2])
    mean_acc = (mean_mat[0] + mean_mat[3]) / np.sum(mean_mat)
    mean_prec = mean_mat[0] / (mean_mat[0] + mean_mat[2])
    mean_f1 = (2 * mean_mat[0]) / (2 * mean_mat[0] + mean_mat[2] + mean_mat[1])
    mean_auroc = aurocs.mean().item()
    mean_g = np.sqrt(mean_spec * mean_sensi)

    # compute mean sensitivity and specificity (with norm class):
    matrix_with_norm = np.vstack((matrix, norm_vec))
    mean_mat_n = np.mean(matrix_with_norm, axis=0)
    mean_sensi_n = mean_mat_n[0] / (mean_mat_n[0] + mean_mat_n[1])
    mean_spec_n = mean_mat_n[3] / (mean_mat_n[3] + mean_mat_n[2])
    mean_acc_n = (mean_mat_n[0] + mean_mat_n[3]) / np.sum(mean_mat_n)
    mean_prec_n = mean_mat_n[0] / (mean_mat_n[0] + mean_mat_n[2])
    mean_f1_n = (2 * mean_mat_n[0]) / (2 * mean_mat_n[0] + mean_mat_n[2] + mean_mat_n[1])
    # mean_auroc = aurocs.mean().item()
    mean_g_n = np.sqrt(mean_spec_n * mean_sensi_n)

    print('Final Test Results with Norm: \n ' + str(matrix) + '\n' + str(norm_vec) + '\n\n' +
          'MI: \n\tsensitivity - ' + str(MI_sensi) + '\n\tspecificity - ' + str(MI_spec) + '\n\tprecision - ' + str(MI_prec) + '\n\taccuracy - ' + str(MI_acc) + '\n\tF1 Score - ' + str(MI_f1) + '\n\tAUROC - ' + str(MI_auroc) + '\n' +
          'STTC: \n\tsensitivity - ' + str(STTC_sensi) + '\n\tspecificity - ' + str(STTC_spec) + '\n\tprecision - ' + str(STTC_prec) + '\n\taccuracy - ' + str(STTC_acc) + '\n\tF1 Score - ' + str(STTC_f1) + '\n\tAUROC - ' + str(STTC_auroc) + '\n' +
          'CD: \n\tsensitivity - ' + str(CD_sensi) + '\n\tspecificity - ' + str(CD_spec) + '\n\tprecision - ' + str(CD_prec) + '\n\taccuracy - ' + str(CD_acc) + '\n\tF1 Score - ' + str(CD_f1) + '\n\tAUROC - ' + str(CD_auroc) + '\n' +
          'HYP: \n\tsensitivity - ' + str(HYP_sensi) + '\n\tspecificity - ' + str(HYP_spec) + '\n\tprecision - ' + str(HYP_prec) + '\n\taccuracy - ' + str(HYP_acc) + '\n\tF1 Score - ' + str(HYP_f1) + '\n\tAUROC - ' + str(HYP_auroc) + '\n' +
          'NORM: \n\tsensitivity - ' + str(NORM_sensi) + '\n\tspecificity - ' + str(NORM_spec) + '\n\tprecision - ' + str(NORM_prec) + '\n\taccuracy - ' + str(NORM_acc) + '\n\tF1 Score - ' + str(NORM_f1) + '\n' +
          'mean: \n\tG-Mean - ' + str(mean_g_n) + '\n\tsensitivity - ' + str(mean_sensi_n) + '\n\tspecificity - ' + str(mean_spec_n) + '\n\tprecision - ' + str(mean_prec_n) + '\n\taccuracy - ' + str(mean_acc_n) + '\n\tF1 Score - ' + str(mean_f1_n))

    print('\n\n Final Test Results without Norm: \n' +
          'mean: \n\tG-Mean - ' + str(mean_g) + '\n\tsensitivity - ' + str(mean_sensi) + '\n\tspecificity - ' + str(mean_spec) + '\n\tprecision - ' + str(mean_prec) + '\n\taccuracy - ' + str(mean_acc) + '\n\tF1 Score - ' + str(mean_f1) + '\n\tAUROC - ' + str(mean_auroc))


if __name__ == '__main__':
    main()
