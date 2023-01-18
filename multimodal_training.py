import pandas as pd

from logger import FhpLogger
from late_fusion import training_late
from early_fusion import training_early
from joint_fusion import training_joint

fhplog = FhpLogger(
    config_file_path="~/Desktop/zuliprc",
    user_id="ricardo.santos@aicos.fraunhofer.pt",
    to=["Logging-RicardoSantos"],
    msg_type="stream",
    topic="default",
)


def central_station(strategy, gpu_id, sig_type, img_type, signal_data, image_data, dropout, batch_size, hidden_size,
                    optimizer, learning_rate, l2_decay, epochs, path_save_model, patience, early_stop, test_id):

    if strategy == "early":
        training_early(gpu_id, sig_type, img_type, signal_data, image_data, dropout, batch_size, hidden_size,
                       optimizer, learning_rate, l2_decay, epochs, path_save_model, patience, early_stop, test_id)

    elif strategy == "late":
        training_late(gpu_id, sig_type, img_type, signal_data, image_data, dropout, batch_size, hidden_size,
                      optimizer, learning_rate, l2_decay, epochs, path_save_model, patience, early_stop, test_id)

    elif strategy == "joint":
        training_joint(gpu_id, sig_type, img_type, signal_data, image_data, dropout, batch_size, hidden_size,
                       optimizer, learning_rate, l2_decay, epochs, path_save_model, patience, early_stop, test_id)


@fhplog.train_logger
def iterator(tests):

    total = len(tests)

    fhplog.send_message("Starting")
    for idx, row in tests.iterrows():
        fhplog.send_message("Starting Test ID: {} | Iteration{}/{}".format(row["test_id"], idx + 1, total))
        print(row.to_dict())

        strategy = row["strategy"]
        dropout = row["dropout"]
        batch_size = row["batch_size"]
        hidden_size = row["hidden_size"]
        optimizer = row["optimizer"]
        learning_rate = row["learning_rate"]
        l2_decay = row["l2_decay"]
        test_id = row["test_id"]

        central_station(strategy, gpu_id, sig_type, img_type, signal_data, image_data, dropout, batch_size, hidden_size,
                        optimizer, learning_rate, l2_decay, epochs, path_save_model, patience, early_stop, test_id)


if __name__ == "__main__":

    tests = pd.read_csv("multimodal_tests_t.csv", delimiter=";", decimal=",")
    tests = tests.sample(frac=1).reset_index(drop=True)

    gpu_id = 0
    sig_type = 'bigru'
    img_type = 'alexnet'
    #signal_data = '/net/sharedfolders/datasets/MOTION/SAIFFER/RicardoPhD/data_for_rnn/'
    signal_data = 'Dataset/data_for_rnn/'
    image_data = 'Dataset/Images/'
    #image_data = '/net/sharedfolders/datasets/MOTION/SAIFFER/RicardoPhD/Images/'
    epochs = 1
    path_save_model = 'save_models/paper_results/'
    patience = 10
    early_stop = True

    iterator(tests)




