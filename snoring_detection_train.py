import os
import sys
import matplotlib
from matplotlib import cm
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, iirnotch
import pandas as pd
from features import FeatureExtractor
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

matplotlib.style.use('ggplot')

# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

def pull_data(dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    x = []
    timestamps = []
    for line in f:
        value = line.split(',')
        if len(value) > 1:
            timestamps.append(float(value[-2]))
            p = float(value[-1])
            x.append(p)
    c = timestamps[0]
    timestamps[:] = [(y - c)/1000 for y in timestamps]
    return np.array(x), np.array(timestamps)


# extract test data for graph
signal, timestamps = pull_data('data', 'audio_snoring_1')
sampling_rate = len(timestamps)/max(timestamps)

"""
plt.figure(figsize=(10,5))
plt.plot(timestamps, signal, 'r-',label='snore_1')
plt.title("Snore Sample 1")
pl.grid()
pl.show()
"""

data_dir = 'data' # directory where the data files are stored
data = np.zeros((0, 2)) # 3 = 1 (timestamp) + 1 (for the audio stream) + 1 (label)
output_dir = 'training_output' # directory where the classifier(s) are stored
class_names = ['snoring', 'none'] # classifiers to train

for filename in os.listdir(data_dir):
	if filename.endswith(".csv") and filename.startswith("audio"):
		filename_components = filename.split("_") # split by the '-' character
		speaker = filename_components[1]
		print("Loading data for {}.".format(speaker))
		if speaker not in class_names:
			class_names.append(speaker)
		speaker_label = class_names.index(speaker)
		sys.stdout.flush()
		data_file = os.path.join(data_dir, filename)
		data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
		print("Loaded {} raw labelled audio data samples.".format(len(data_for_current_speaker)))
		sys.stdout.flush()
		data = np.append(data, data_for_current_speaker, axis=0)

print("Found data for {} speakers : {}".format(len(class_names), ", ".join(class_names)))




# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

n_features = 1062

print("Extracting features and labels for {} audio windows...".format(data.shape[0]))
sys.stdout.flush()

X = np.zeros((0,n_features))
y = np.zeros(0,)

# change debug to True to show print statements we've included:
feature_extractor = FeatureExtractor(debug=True) 

nr_total_windows = 0
nr_bad_windows = 0
nr_windows_with_zeros = 0
print("okay before for loop")
n = 0
for i,window_with_timestamp_and_label in enumerate(data):
    window = window_with_timestamp_and_label[1:-1]
    label = data[i,-1]
    nr_total_windows += 1
    try:
        x = feature_extractor.extract_features(window)
        print("completed feature extractor")
        print(len(x))
        if (len(x) != X.shape[1]):
            print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))
        X = np.append(X, np.reshape(x, (1,-1)), axis=0)
        y = np.append(y, label)
    except:
        nr_bad_windows += 1
        if np.all((window == 0)):
            nr_windows_with_zeros += 1
print("{} windows found".format(nr_total_windows))
print("{} bad windows found, with {} windows with only zeros".format(nr_bad_windows, nr_windows_with_zeros))
