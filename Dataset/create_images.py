## Create the dataset to train a CNN (AlexNet) to perform ECG classification 
## Based on the GAF, RP and MTF transformations applied to the 1D ECG

#filter the ecg signal (band pass filter)
#select 3 leads (I, II, V2)
#convert to images (9,1000,1000) for each example

#save the images as tiff files  (0.tif to number_of_examples.tif) 
#save the labels as numpy arrays (0.tif to number_of_examples.tif)



import cv2
import tifffile as tf
import numpy as np

a = np.random.randint(1,255,(9,1000,1000))

a = a.astype('uint8')

tf.imwrite(r'C:\Users\hemax\Desktop\0.tif', a)

 import numpy as np

a = np.zeros(4)

a[1] =1

a[3] = 1

np.save(r'C:\Users\hemax\Desktop\0.npy', a)


#band pass filter
from scipy.signal import butter, sosfilt

band_pass_filter = butter(2, [0.05, 40], 'bandpass', fs=100, output='sos' )

filtered_signal = sosfilt(band_pass_filter, signal)


normalize 

x_example_lead1 = (x_example_lead1 -min(x_example_lead1)) / max(x_example_lead1-min(x_example_lead1))
