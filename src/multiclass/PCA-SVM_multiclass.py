# -*- coding: utf-8 -*-
"""
Created on Tue May 11 01:03:16 2021

@author: rbarb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:27:04 2021

@author: rbarb
"""
# ANN
# Author: Robert Barbulescu

# importing necessary packages 

from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, roc_curve, cohen_kappa_score
import sklearn.metrics as metrics


# used for better plots
plt.style.use('ggplot')

# The features existing on the dataset 
# The features names were obtained from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
features = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# Mounting and reaidng the csv file with the dataset
# Also available using sklearn packgae: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html
dataset = pd.read_csv(r"C:\Users\rbarb\Desktop\ANN\kddcup.data_10_percent_corrected", names = features, header=None)

# The number of samples and dimensions
print('Data Points:',dataset.shape[0])
print('Features:',dataset.shape[1])
print("Initial data shape is:", dataset.shape)
print("=======================================================================")

# remove any existing duplicates
dataset.drop_duplicates(subset=None, keep='first', inplace=True)
print("Duplicates succesfully dropped.")
print("Dataset shape is:", dataset.shape)
print("=======================================================================")

# Looking for any NULL values
print('Null values existing: ',len(dataset[dataset.isnull().any(1)]))
print("=======================================================================")


# Grouping the values into categories: probe, r2l, u2r and dos
dataset.replace(to_replace = ['ipsweep.', 'portsweep.', 'nmap.', 'satan.'], value = 'Probe', inplace = True)
dataset.replace(to_replace = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'], value = 'R2l', inplace = True)
dataset.replace(to_replace = ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'], value = 'U2r', inplace = True)
dataset.replace(to_replace = ['back.', 'land.' , 'neptune.', 'pod.', 'smurf.', 'teardrop.'], value = 'DoS', inplace = True)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 41].values

# Encoding categorical data of 3 fields: protocol_type, service, and flag
encoder = LabelEncoder()
x[:, 1] = encoder.fit_transform(x[:, 1])
x[:, 2] = encoder.fit_transform(x[:, 2])
x[:, 3] = encoder.fit_transform(x[:, 3])

y = encoder.fit_transform(y)

# Display newly structured classes
print(dataset.groupby('label')['label'].count())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
print("Size of training Data is % d and Testing Data is % d" %(
        y_train.shape[0], y_test.shape[0]))
print("=======================================================================")

# Feature Scalling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Plotting the variance 
pca = PCA().fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance')

plt.show()

pca = PCA(copy=True, iterated_power='auto', n_components=30, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False).fit(x_train)

x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

t0 = time()

print("Sample Data point after applying PCA\n", x_train_pca[0])
print("=======================================================================")
print("Dimesnsions of training set = % s and Test Set = % s"%(
        x_train.shape, x_test.shape))      
print("Fitting the classifier to the training set")
print("=======================================================================")
clf = SVC(kernel='rbf',class_weight="balanced", gamma = "auto")
clf = clf.fit(x_train_pca, y_train)
print("Done in %0.3fs" % (time() - t0))
y_pred = clf.predict(x_test_pca)
print("=======================================================================")
print("Accuracy score:{:.2f}%".format(metrics.accuracy_score(y_test, y_pred)*100))
print("=======================================================================")
print("Number of mislabeled points out of a total %d points is %d" % (x_test.shape[0], (y_test != y_pred).sum()))
print("=======================================================================")

target_names = ['DoS', 'Probe', 'R2l', 'U2r', 'Normal']
# print classifiction results
print(classification_report(y_test, y_pred, target_names=target_names))
# print confusion matrix
print("=======================================================================")
print("Confusion Matrix is:")
print(confusion_matrix(y_test, y_pred))
print("=======================================================================")
