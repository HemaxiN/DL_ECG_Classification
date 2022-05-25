# DL_ECG_Classification

Practical project to compare how different methods for ECG signal representation perform in ECG classification; and to explore a multimodal DL approach to fuse the two models, leveraging the different structures of signal representations.


# Dataset

Details regarding the dataset are presented [here](https://github.com/HemaxiN/DL_ECG_Classification/tree/main/Dataset).

Examples of the ECG signal obtained with leads I, II and V2 for the ECG record 17110 (with ground truth label NORM).

![](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Images/ecg_record_17110.PNG)



# Image Sub-Net: CNN based on the AlexNet Model for ECG classification

Examples of GAF (left), MTF (middle) and RP (right) images for the ECG record 17110 (with ground truth label NORM), corresponding to leads I, II and V2:

![](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Images/examples_GAF_MTF_RP1.png)

To obtain the images shown above run the file [create_images.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Dataset/create_images.py), specifying the partition ('train', 'dev' or 'test') and the directory containing the [processed files](https://drive.google.com/drive/folders/1Nas7Gqcj-H28Raui_6z06kpWDsM78OBV), and the directory where the images and corresponding labels will be saved.

This will create a directory with the training, validation and test sets with the following tree structure:

```
train_dev_test_dataset
├── train
│   ├── images
│   └── labels
└── dev
│   ├── images
│   └── labels
└── test
    ├── images
    └── labels
```

To train the model use the following command (correctly specifying the directory of the dataset (using ```-data```) and the directory to save the model (using ```-path_save_model```). Other parameters can be specified as explained [here](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/AlexNet.py#L145-L157).

```
python3 AlexNet.py -data '/dev/shm/dataset' -epochs 100 -batch_size 64 -path_save_model '/mnt/2TBData/hemaxi/ProjetoDL/working' -gpu_id 0 -learning_rate 0.1  
```

![](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Images/CNN.png)

The models that we trained are available [here](https://drive.google.com/drive/folders/12LVLYfjd2N4mTuMwAmtJ6wOwm3y9HWAT?usp=sharing), (alexnet and resnet denote respectively the best alexnet and resnet models based on their performance on the validation set).

Configuration of the CNN based on the AlexNet model. nb_filters denotes the number of filters in the first layer used to compute the number of filters in the following layers; batch denotes the batch size:
```
|           Layer         |                   Size                 |
|:-----------------------:|:--------------------------------------:|
|     Convolutional 2D    |        (batch, nb_filters, 62,62)      |
|           ReLU          |       (batch, nb_filters, 62, 62)      |
|       MaxPooling 2D     |        (batch, nb_filters, 30,30)      |
|        Dropout 2D       |        (batch, nb_filters, 30,30)      |
|     Convolutional 2D    |       (batch, nb_filters×2, 30,30)     |
|           ReLU          |     (batch,  nb_filters×2,   30,30)    |
|       MaxPooling 2D     |     (batch,  nb_filters×2,   14,14)    |
|        Dropout 2D       |     (batch,  nb_filters×2,   14,14)    |
|     Convolutional 2D    |     (batch,  nb_filters×4,   14,14)    |
|           ReLU          |     (batch,  nb_filters×4,   14,14)    |
|     Convolutional 2D    |     (batch,  nb_filters×8,   14,14)    |
|           ReLU          |       (batch, nb_filters×8, 14,14)     |
|     Convolutional 2D    |           (batch, 256, 14,14)          |
|           ReLU          |           (batch, 256, 14,14)          |
|       MaxPooling 2D     |            (batch, 256, 6,6)           |
|        Dropout 2D       |            (batch, 256, 6,6)           |
|          Linear         |              (batch, 4096)             |
|           ReLU          |              (batch, 4096)             |
|          Linear         |              (batch, 2048)             |
|           ReLU          |              (batch, 2048)             |
|          Linear         |               (batch,  4)              |
``` 

To evaluate the performance of the model run the file [load_alexnet_evaluate.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/load_alexnet_evaluate.py), specifying the directories of the trained model and dataset. This will output a matrix of dimension (4×4) with the true positives (TP), false negatives (FN), false positives (FP) and true negatives (TN) for each class separately:

```
|      | TP | FN | FP | TN |
|------|----|----|----|----|
| MI   |    |    |    |    |
| STTC |    |    |    |    |
| CD   |    |    |    |    |
| HYP  |    |    |    |    |
```

# Acknowledgements

* The signal to image transformation methods (implemented in [create_images.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Dataset/create_images.py) and [ecgtoimage.ipynb](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Dataset/ecgtoimage.ipynb)) are based on the publicly available implementation by [Ahmad et al., 2021, ECG Heartbeat Classification Using Multimodal Fusion](https://github.com/zaamad/ECG-Heartbeat-Classification-Using-Multimodal-Fusion)
* The structure of the [AlexNet.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/AlexNet.py) and [cnn.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/cnn.py) is based on the source code of [Homework 1 and 2 of the Deep Structured Learning Course](https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks).
* ResNet code based on: https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py

