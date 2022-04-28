PTB-XL, a large publicly available electrocardiography dataset (available at https://physionet.org/content/ptb-xl/1.0.1/) 
 
The training examples and corresponding labels are obtained running the "dataset.ipynb" notebook (https://drive.google.com/drive/folders/1h-LlDgo7bQ_Xg-wrO7_3XG3uAjWAAyl9?usp=sharing)
 
X_train.pickle (19634, 1000, 12) 

y_train.pickle (19634,) 
 
X_test.pickle (2203, 1000, 12) 

y_test.pickle (2203,) 
 
How to load the data 

pickle_in = open("name.pickle","rb") 

name = pickle.load(pickle_in)
