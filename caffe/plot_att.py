import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
import numpy as np
accuracy = []
classifier_file = '../pkl/accuracy_obj_file.pkl'
pkl_file = open(classifier_file, 'rb')
accuracy = pickle.load(pkl_file)
#print(accuracy)
index = []
for i in range(len(accuracy)) :
	index.append(i)
 

print(np.average(accuracy))
plt.bar(index, accuracy, align='center', alpha=0.5)
plt.title('Accuracy of object''s linear classifier')
plt.xlabel('Object Id''s') 
plt.ylabel('Accuracy')
plt.show()

