# DL_ECG_Classification

Official implementation of the paper: https://www.sciencedirect.com/science/article/pii/S174680942400199X

Practical project to compare how different methods for ECG signal representation perform in ECG classification; and to explore a multimodal DL approach to fuse the two models, leveraging the different structures of signal representations.


# Dataset

Details regarding the dataset are presented [here](https://github.com/HemaxiN/DL_ECG_Classification/tree/main/Dataset).

Examples of the ECG signal obtained with leads I, II and V2 for the ECG record 17110 (with ground truth label NORM).

![](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Images/ecg_record_17110.PNG)



# Image Sub-Net: CNNs for ECG classification

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

To train the model use the following command (selecting the model ```AlexNet.py```, ```resnet.py```, ```vggnet.py``` or ```alexnetattention.py```, and correctly specifying the directory of the dataset (using ```-data```) and the directory to save the model (using ```-path_save_model```). Other parameters can be specified as explained [here](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/AlexNet.py#L145-L157).
For instance:

```
python3 AlexNet.py -data '/dev/shm/dataset' -epochs 100 -batch_size 256 -path_save_model '/mnt/2TBData/hemaxi/ProjetoDL/working' -gpu_id 0 -learning_rate 0.01  
```

Please note that the optimized configurations for each architecture are based on the findings presented in our [paper](https://www.sciencedirect.com/science/article/pii/S174680942400199X). We recommend using the parameters specified below and referring to the paper for further details:

* (a) AlexNet: #filters=16, a batch size of 256 (adjust as needed based on GPU memory), a learning rate of 0.01 and a dropout rate of 0;
* (b) ResNet: #filters=16, batch size of 128 (adjust as needed based on GPU memory), a learning rate of 0.01; 
* (c) VGGNet: #filters of 16, a batch size of 128 (adjust as needed based on GPU memory), a learning rate of 0.1 and a dropout rate of 0.3; 
* (d) MobileNetV2: #filters of 32, a batch size of 16 (adjust as needed based on GPU memory), a learning rate of 0.1 and a dropout rate of 0; 
* (e) AlexNetAtt: #filters of 8, a batch size of 16 (adjust as needed based on GPU memory), a learning rate of 0.01 and a dropout rate of 0.



![](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Images/cnn_conf2.png)

The models that we trained are available [here](https://drive.google.com/file/d/1uRz6RsfitbCNzf5Z1-H82GvkjdmNWWDC/view?usp=share_link), (alexnet, resnet, vggnet and customcnn denote respectively the best AlexNet, ResNet, VGGNet and custom CNN models based on their performance on the validation set).

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
* The structure of the [AlexNet.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/AlexNet.py), [ResNet.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/resnet.py), [VGGNet.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/vggnet.py) and [cnn.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/cnn.py) is based on the source code of [Homework 1 and 2 of the Deep Structured Learning Course](https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks).
* AlexNet code based on: https://medium.com/analytics-vidhya/alexnet-a-simple-implementation-using-pytorch-30c14e8b6db2 (visited on April 27, 2022)
* ResNet code based on: https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py (visited on May 22, 2022)
* VGGNet code based on: https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1 (visited on May 22, 2022)
