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


# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

print("\n")
print("---------------------- Decision Tree -------------------------")

total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0, 0.0]

cv = KFold(n_splits=10, shuffle=True, random_state=None)
for i, (train_index, test_index) in enumerate(cv.split(X)):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
	print("Fold {} : Training decision tree classifier over {} points...".format(i, len(y_train)))
	sys.stdout.flush()
	tree.fit(X_train, y_train)
	print("Evaluating classifier over {} points...".format(len(y_test)))

	# predict the labels on the test data
	y_pred = tree.predict(X_test)

	# show the comparison between the predicted and ground-truth labels
	conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])

	accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
	precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
	recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

	total_accuracy += accuracy
	total_precision += precision
	total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  

print("Training decision tree classifier on entire dataset...")
tree.fit(X, y)

print("\n")
print("---------------------- Random Forest Classifier -------------------------")
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0, 0.0]

for i, (train_index, test_index) in enumerate(cv.split(X)):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	print("Fold {} : Training Random Forest classifier over {} points...".format(i, len(y_train)))
	sys.stdout.flush()
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, y_train)

	print("Evaluating classifier over {} points...".format(len(y_test)))
	# predict the labels on the test data
	y_pred = clf.predict(X_test)

	# show the comparison between the predicted and ground-truth labels
	conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])

	accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
	precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
	recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

	total_accuracy += accuracy
	total_precision += precision
	total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  

# TODO: (optional) train other classifiers and print the average metrics using 10-fold cross-validation

# Set this to the best model you found, trained on all the data:
best_classifier = RandomForestClassifier(n_estimators=100)
best_classifier.fit(X,y) 

classifier_filename='classifier.pickle'
print("Saving best classifier to {}...".format(os.path.join(output_dir, classifier_filename)))
with open(os.path.join(output_dir, classifier_filename), 'wb') as f: # 'wb' stands for 'write bytes'
	pickle.dump(best_classifier, f)