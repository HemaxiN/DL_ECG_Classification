# DL_ECG_Classification

Practical project to explore a multimodal deep learning approach for ECG classification. 


# Dataset

Details regarding the dataset are presented [here](https://github.com/HemaxiN/DL_ECG_Classification/tree/main/Dataset).

Examples of GAF (left), MTF (middle) and RP (right) images corresponding to a record with ground truth label NORM, for leads I, II and V2:

![](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Images/examples_GAF_MTF_RP.png)

# Running the AlexNet Model for ECG classification

```python3 AlexNet.py -data '/mnt/2TBData/hemaxi/DL/projeto/ds' -epochs 100 -batch_size 2 -path_save_model '/mnt/2TBData/hemaxi/DL/projeto' -gpu_id 0```


# Acknowledgements

* The signal to image transformation methods (implemented in [create_images.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Dataset/create_images.py) and [ecgtoimage.ipynb](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Dataset/ecgtoimage.ipynb)) are based on the publicly available implementation by [Ahmad et al., 2021, ECG Heartbeat Classification Using Multimodal Fusion](https://github.com/zaamad/ECG-Heartbeat-Classification-Using-Multimodal-Fusion)
* The structure of the [AlexNet.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/AlexNet.py) and [cnn.py](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/cnn.py) is based on the source code of [Homework 1 and 2 of the Deep Structured Learning Course](https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks).
