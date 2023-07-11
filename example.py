import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,LeaveOneOut
from random import randint

#prepare the RandomForestClassifier, used as is.
X = np.load('dataset_X.npy')
y = np.load('dataset_y.npy')
clf = RandomForestClassifier(n_estimators=100,min_samples_leaf=2)
clf.fit(X,y)

#randomly pick a instance from dataset and give a prediction
idx = randint(0,len(X))
vec = X[idx]
true_y = y[idx]
pred_y = clf.predict([vec])[0]

#print the prediction
print(f"Selected idx:{idx}\nfeature vector:{vec}\ntrue_type:{true_y}\tpredicted_type:{pred_y}")