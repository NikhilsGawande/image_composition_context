import numpy as np
import pickle
from sklearn import svm 
from sklearn.metrics import accuracy_score
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import im_utils
"""
x_train = np.loadtxt('features.txt', dtype=float)
y_train = np.loadtxt('labels.txt', dtype=float)

print(x_train)
print(y_train)
"""
classifier = []
accuracy = []
features_all = '../pkl/non_att_features/features_all.pkl'
pkl_file = open(features_all, 'rb')
x_neg = pickle.load(pkl_file)
m = im_utils.load('./mitstates_data/split_meta_info.pklz')
#Extracting positive features 
no_of_files = 10
for i in range(no_of_files) :
	file_name = '../pkl/att_features/features_' + str(i) + '.pkl'
	pkl_file1 = open(file_name, 'rb')
	x_pos = pickle.load(pkl_file1)
	att_len = 115
	#Extracting negative features 
	x_negative = []
	x_pos_len = len(x_pos)
	mul = int(x_pos_len/att_len)
	mul = mul + 1
	for j in range(att_len) :
		if j == i :
			continue
		for k in range(mul) :
			x_negative.append(x_neg[j][k])
	x_positive = x_pos.tolist()
	x_pos_train = x_positive[:int(0.7*x_pos_len)]
	x_pos_test = x_positive[int(0.7*x_pos_len):]
	x_neg_train = x_negative[:int(0.7*x_pos_len)]
	x_neg_test = x_negative[int(0.7*x_pos_len):]
	x_train = x_pos_train + x_neg_train
	x_test = x_pos_test + x_neg_test
	y_train = []
	one = 1
	minusone = -1
	for i in range(len(x_pos_train)) :
		y_train.append(one)

	for i in range(len(x_neg_train)) :
		y_train.append(minusone)

	clf = svm.LinearSVC()
	#x_train = x_train.reshape(1,-1)
	#y_train = y_train.reshape(1,-1)
	clf.fit(x_train, y_train)  
	classifier.append(clf.coef_.tolist())
	y_true = []
	for i in range(len(x_pos_test)) :
		y_true.append(one)
	for i in range(len(x_neg_test)):
		y_true.append(minusone)
	y_pred = []
	for a in x_test :
		b = np.array(a, ndmin=2)
		y_pred.append(clf.predict(b)[0])
	print(accuracy_score(y_true, y_pred))
	accuracy.append(accuracy_score(y_true, y_pred))
	del(x_pos)
	del(x_positive[:])
	del(x_negative[:])
	del(x_pos_train[:])
	del(x_neg_train[:])
	del(x_pos_test[:])
	del(x_neg_test[:])
	del(x_train[:])
	del(x_test[:])
	del(y_train[:])
	del(y_true[:])
	del(y_pred[:])

classifier_file = '../pkl/classifier_att_file.pkl'
pkl_file = open(classifier_file, 'wb')
pickle.dump(classifier, pkl_file)
pkl_file.close()

accuracy_file = '../pkl/accuracy_att_file.pkl'
pkl_file = open(accuracy_file, 'wb')
pickle.dump(accuracy, pkl_file)
pkl_file.close()
index = []
for i in range(no_of_files) :
	index.append(i)
 
num_bins = 5

plt.bar(index, accuracy, align='center', alpha=0.5)
plt.title('Accuracy of attribute''s linear classifier')
plt.xlabel('Attribute Id''s') 
plt.ylabel('Accuracy')
plt.show()

