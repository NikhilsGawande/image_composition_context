import numpy as np
import pickle
from sklearn import svm 
"""
x_train = np.loadtxt('features.txt', dtype=float)
y_train = np.loadtxt('labels.txt', dtype=float)

print(x_train)
print(y_train)
"""

pkl_file1 = open('../pkl/att_features/features_0.pkl', 'rb')
x_train = pickle.load(pkl_file1)
#print(x_train)


pkl_file2 = open('labels.pkl', 'rb')
y_train = pickle.load(pkl_file2)
#print(y_train)



clf = svm.LinearSVC()
#x_train = x_train.reshape(1,-1)
#y_train = y_train.reshape(1,-1)
clf.fit(x_train, y_train)  
#print(clf.coef_)

pkl_file3 = open('predict_positive.pkl', 'rb') 
x_test = pickle.load(pkl_file3)

pkl_file4 = open('predict_negative.pkl', 'rb') 
y_test = pickle.load(pkl_file4)
print("positive")
for a in x_test:
	b = np.array(a, ndmin=2)
	print(clf.predict(b))

print("negative")	
for a in y_test : 
	b = np.array(a, ndmin=2)
	print(clf.predict(b))

