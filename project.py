import matplotlib
from matplotlib import cm
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

matplotlib.style.use('ggplot')

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

headers = ['Name', 'Age', 'Marks']

df = pd.read_csv("./data/Snoring-2.csv", header=None, squeeze=False)[:20_000]
df.plot()

plt.show()



