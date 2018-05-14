import caffe
import numpy as np
import os 
#from sklearn import svm 
import pickle
import sys 
sys.path.insert(0, '../database/')
import read_data_list 

#Setup VGG-1-M-1024
#caffe model setup
net = caffe.Net('./deploy.prototxt',
	'./VGG_CNN_M_1024.caffemodel', caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('./out.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1,3,224,224)

layer = 'fc7'
if layer not in net.blobs:
    raise TypeError("Invalid layer name: " + layer)

#Initializations 

non_attr_temp = []
y_train_temp = []
minusone = -1


#Extracting Images list for atrributes 
non_att_list = read_data_list.read_random_att()

counter = 0
for att in non_att_list :

	for imagename in att :		
		imgname = './../images/' + imagename
		img = caffe.io.load_image(imgname)
		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = net.forward()	
		non_attr_temp.append((net.blobs[layer].data[0].tolist())[:])
		y_train_temp.append(minusone)
		print(counter)
		counter += 1
	print("non_att_list")

	print("features extracted")   
	x_train = np.array(non_attr_temp)
	y_train = np.array(y_train_temp)


	feature_file = '../pkl/non_att_features/features_' + str(non_att_list.index(att)) + '.pkl'
	features = open(feature_file, 'wb')
	pickle.dump(x_train, features)
	features.close()

	label_file = '../pkl/non_att_features/labels_' + str(non_att_list.index(att)) + '.pkl'
	labels = open(label_file, 'wb')
	pickle.dump(y_train, labels) 
	labels.close()

	print("features stored")
	del(non_attr_temp[:])
	del(y_train_temp[:])
	del(x_train)
	del(y_train)
	counter = 0