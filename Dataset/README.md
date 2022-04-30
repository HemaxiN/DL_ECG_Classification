# PTB-XL, a large publicly available electrocardiography dataset (available at [Physionet](https://physionet.org/content/ptb-xl/1.0.1/))
 
The training examples and corresponding labels are obtained running the [dataset.ipynb](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Dataset/dataset.ipynb). 
 
```
X_train_processed.pickle (17111, 1000, 12)
y_train_processed.pickle (17111, 4)
X_dev_processed.pickle (2156, 1000, 12)
y_dev_processed.pickle (2156, 4)
X_test_processed.pickle (2163, 1000, 12)
y_test_processed.pickle (2163, 4)
```

These are saved as numpy arrays and available [here](https://drive.google.com/drive/folders/1Nas7Gqcj-H28Raui_6z06kpWDsM78OBV?usp=sharing).
 
# Encoding 

label_encoding (shown in function labelstovector in [this notebook](https://github.com/HemaxiN/DL_ECG_Classification/blob/main/Dataset/dataset.ipynb))

Multi-label problem:

```
Norm: [0,0,0,0]
MI: [1,0,0,0]
STTC: [0,1,0,0]
CD: [0,0,1,0]
HYP: [0,0,0,1]
Label combination example:
HYP and MI: [1,0,0,1]
HYP and CD and STTC: [0,1,1,1]
```

# How to load the data 

```
pickle_in = open("name.pickle","rb") 
name = pickle.load(pickle_in)
```
