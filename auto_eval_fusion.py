import torch
import os

from torch.utils.data import DataLoader
import gru as gru
import alexnetattention as alexnet
import late_fusion as late
import early_fusion as early
import joint_fusion as joint

from utils import compute_save_metrics_with_norm
from count_parameters import count_parameters

import os
import fnmatch
import pandas as pd

def list_all_models(directory, excluded_formats=["txt", "pdf", "csv", "xlsx"]):

    files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            if not any(fnmatch.fnmatch(file, f'*.{fmt}') for fmt in excluded_formats):
                #if pd.Timestamp(file.split("_")[2]) >= pd.Timestamp(year=2023, month=12, day=1):
                files.append(file)
    
    files = ["late_model_2023-11-24_17-45-47_lr0.001_optadam_dr0_eps200_hs128_bs256_l20"]

    df = pd.DataFrame(columns=["file", "strategy", "specific", "lr", "hs", "bs", "attention"])
    for file in files:

        split = file.split("_")

        if len(split) == 13:

            # needs to skip wrong joint model tests
            if split[0] == "joint" and split[-1] != "3":
                print("SKIP:", file)
                continue
            new_data = [file, split[0], split[-2] + "_" + split[-1], split[-9][2:], split[-5][2:], split[-4][2:], "True"]
            df = pd.concat([df, pd.DataFrame([new_data], columns=df.columns)], ignore_index=True)
        
        elif len(split) == 14:
            new_data = [file, split[0], split[-3] + "_" + split[-2], split[-10][2:], split[-6][2:], split[-5][2:], split[-1]]
            df = pd.concat([df, pd.DataFrame([new_data], columns=df.columns)], ignore_index=True)
        
        elif split[0] == "early":
            new_data = [file, split[0], "conv2d_5", split[-7][2:], split[-3][2:], split[-2][2:], "True"]
            df = pd.concat([df, pd.DataFrame([new_data], columns=df.columns)], ignore_index=True)
        
        elif split[0] == "joint":
            new_data = [file, split[0], "layer_3", split[-7][2:], split[-3][2:], split[-2][2:], "True"]
            df = pd.concat([df, pd.DataFrame([new_data], columns=df.columns)], ignore_index=True)
        
        else:
            new_data = [file, split[0], "", split[-7][2:], split[-3][2:], split[-2][2:], "False"]
            df = pd.concat([df, pd.DataFrame([new_data], columns=df.columns)], ignore_index=True)
    
    df["lr"] = df["lr"].astype(float)
    df["bs"] = df["bs"].astype(int)
    df["hs"] = df["hs"].astype(int)
    df["attention"] = df["attention"].map({"True": True, "False": False})

    return df


def run_tests(test):
    sig_model = gru.RNN(3, 128, 3, 4, 0, gpu_id=gpu_id, bidirectional=False).to(gpu_id)
    img_model = alexnet.AlexNet(4).to(gpu_id)

    if test["strategy"] == "late":
        sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
        sig_model.eval()
        img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
        img_model.eval()

        val_dataset = late.LateFusionDataset(signal_data, image_data, sig_model, img_model, 'gru', 'alexnet',
                                             dataset_size, gpu_id, 1, part='dev')

        test_dataset = late.LateFusionDataset(signal_data, image_data, sig_model, img_model, 'gru', 'alexnet',
                                             dataset_size, gpu_id, 1, part='test')

        model = late.LateFusionNet(4, 8, test["hs"], 0, test["attention"]).to(gpu_id)
    
    elif test["strategy"] == "early":
        sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
        sig_model.requires_grad_(False)
        sig_model.eval()
        sig_model.rnn.register_forward_hook(early.get_activation('rnn'))
        
        img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
        img_model.requires_grad_(False)
        img_model.eval()
        img_model.__getattr__(test["specific"]).register_forward_hook(early.get_activation(test["specific"]))

        val_dataset = early.FusionDataset(signal_data, image_data, dataset_size, part='dev')
        test_dataset = early.FusionDataset(signal_data, image_data, dataset_size, part='test')

        img_size = {'conv2d_1': 0, 'conv2d_2': 0, 'conv2d_3': 512, 'conv2d_4': 1024, 'conv2d_5': 4096}
        img_features = img_size[test["specific"]]

        model = early.EarlyFusionNet(4, 128, img_features, test["hs"], 0,
                                     sig_model, img_model, "rnn", test["specific"], test["attention"]).to(gpu_id)
    
    else:
        sig_model.load_state_dict(torch.load(sig_path, map_location=torch.device(gpu_id)))
        img_model.load_state_dict(torch.load(img_path, map_location=torch.device(gpu_id)))
        
        sig_model.fc = joint.Identity()
        sig_features = 128
        
        img_model.linear_3 = joint.Identity()  # applied on the last dense layer only
        img_features = 2048
        
        if test["specific"] == "layer_2":
            img_model.linear_2 = joint.Identity()
            img_features = 4096
        
        if test["specific"] == "layer_1":
            img_model.linear_2 = joint.Identity()
            img_model.linear_1 = joint.Identity()
            img_features = 9216
        
        val_dataset = early.FusionDataset(signal_data, image_data, dataset_size, part='dev')
        test_dataset = early.FusionDataset(signal_data, image_data, dataset_size, part='test')

        model = joint.JointFusionNet(4, sig_features, img_features, test["hs"], 0, sig_model, img_model, test["attention"]).to(gpu_id)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.load_state_dict(torch.load(dir_models + os.sep + test["file"], map_location=torch.device(gpu_id)))
    model = model.to(gpu_id)
    model.eval()

    class_weights = torch.tensor([17111/4389, 17111/3136, 17111/1915, 17111/417], dtype=torch.float)
    class_weights = class_weights.to(gpu_id)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # evaluate the performance of the model
    if test["strategy"] == 'late':
        validation_loss = gru.compute_loss(model, val_dataloader, criterion, gpu_id=gpu_id)
        test_loss = gru.compute_loss(model, test_dataloader, criterion, gpu_id=gpu_id)

        opt_threshold = late.threshold_optimization(model, val_dataloader, gpu_id=gpu_id)

        matrix_val, norm_val = gru.evaluate_with_norm(model, val_dataloader, opt_threshold, gpu_id=gpu_id)
        aurocs_val = gru.auroc(model, val_dataloader, gpu_id=gpu_id)

        matrix, norm = gru.evaluate_with_norm(model, test_dataloader, opt_threshold, gpu_id=gpu_id)
        aurocs = gru.auroc(model, test_dataloader, gpu_id=gpu_id)

    else:
        validation_loss = early.fusion_compute_loss(model, val_dataloader, criterion, gpu_id=gpu_id)
        test_loss = early.fusion_compute_loss(model, test_dataloader, criterion, gpu_id=gpu_id)

        opt_threshold = early.fusion_threshold_optimization(model, val_dataloader, gpu_id=gpu_id)

        matrix_val, norm_val = early.fusion_evaluate_with_norm(model, val_dataloader, opt_threshold, gpu_id=gpu_id)
        aurocs_val = early.fusion_auroc(model, val_dataloader, gpu_id=gpu_id)
        
        matrix, norm = early.fusion_evaluate_with_norm(model, test_dataloader, opt_threshold, gpu_id=gpu_id)
        aurocs = early.fusion_auroc(model, test_dataloader, gpu_id=gpu_id)

    val_dict, test_dict = compute_save_metrics_with_norm(matrix, norm, aurocs, test_loss, matrix_val, norm_val, aurocs_val, validation_loss)

    return val_dict, test_dict, count_parameters(model)

if __name__ == "__main__":

    local = False
    dataset_size = [1500, 500, 500] if local else [17111, 2156, 2163]
    signal_data = 'Dataset/short_data_for_rnn/' if local else 'Dataset/data_for_rnn/'
    image_data = 'Dataset/short_Images/' if local else 'Dataset/Images/'
    
    dir_models = "save_models/paper_results_revision"

    gpu_id = 0 if torch.cuda.is_available() else "cpu"

    tests_df = list_all_models(dir_models)

    sig_path = "best_trained_rnns/gru_3layers_dropout0_model8"
    img_path = "save_models/alexnetatt"

    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    params_df = pd.DataFrame()
    for i, test in tests_df.iterrows():
        print("\n\n")
        print(test.to_dict())

        val_dict, test_dict, params = run_tests(test)

        val_df = pd.concat([val_df, pd.DataFrame([val_dict], index=[test['file']])], ignore_index=False)
        test_df = pd.concat([test_df, pd.DataFrame([test_dict], index=[test['file']])], ignore_index=False)
        params_df = pd.concat([params_df, pd.DataFrame([{"Parameters": params}], index=[test['file']])], ignore_index=False)

    with pd.ExcelWriter(dir_models + os.sep + "final_results_t.xlsx", engine='openpyxl') as writer:
        val_df.to_excel(writer, sheet_name='Validation')
        test_df.to_excel(writer, sheet_name='Test')
        params_df.to_excel(writer, sheet_name='Parameters')
    