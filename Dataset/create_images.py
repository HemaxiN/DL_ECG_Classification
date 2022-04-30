## Create the dataset to train a CNN (AlexNet) to perform ECG classification 
## Based on the GAF, RP and MTF transformations applied to the 1D ECG

#filter the ecg signal (band pass filter)
#select 3 leads (I, II, V2)
#convert to images (9,1000,1000) for each example

#save the images as tiff files  (0.tif to number_of_examples.tif)
#save the labels as numpy arrays (0.tif to number_of_examples.tif)
