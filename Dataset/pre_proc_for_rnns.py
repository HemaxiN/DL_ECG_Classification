# Pre-processing of data (X) to be used in the RNNs:
#   1) filter the ecg signal (band pass filter)
#   2) select 3 leads (I, II, V2)
#   3) normalize

import numpy as np
import pickle
import os
from scipy.signal import butter, sosfilt
from scipy.stats import zscore


def X_for_RNNs(path, partition='train', save_dir=None):

	file = str(path) + '/X_' + str(partition) + '_processed.pickle'
	pickle_in = open(file, "rb")
	X = pickle.load(pickle_in)

	# band pass filter
	band_pass_filter = butter(2, [1, 45], 'bandpass', fs=100, output='sos')

	X_aux = np.zeros((np.shape(X)[0], np.shape(X)[1], 3))

	for i in range(np.shape(X)[0]):

		lead_I = X[i][:, 0]  # X[i]: (1000, 12)
		lead_II = X[i][:, 1]
		lead_V2 = X[i][:, 7]

		# apply a band pass filter (0.05, 40hz)
		lead_I = sosfilt(band_pass_filter, lead_I)
		lead_II = sosfilt(band_pass_filter, lead_II)
		lead_V2 = sosfilt(band_pass_filter, lead_V2)

		# normalize before transforming into images
		lead_I = ecgnorm(lead_I)
		lead_II = ecgnorm(lead_II)
		lead_V2 = ecgnorm(lead_V2)

		X_aux[i][:, 0] = lead_I
		X_aux[i][:, 1] = lead_II
		X_aux[i][:, 2] = lead_V2

		np.save(str(save_dir) + '/X_rnn_' + str(partition) + '/' + str(i) + '.npy', X_aux[i])

	return


def ecgnorm(ecg):
	ecg_norm = zscore(ecg)
	return ecg_norm


def labels(path, partition='train', save_dir=None):

	file = str(path) + '/y_' + str(partition) + '_processed.pickle'
	pickle_in_y = open(file, "rb")
	y = pickle.load(pickle_in_y)
	for i in range(np.shape(y)[0]):
		y_i = y[i]
		np.save(str(save_dir) + '/labels_' + str(partition) + '/' + str(i) + '.npy', y_i)
	return


