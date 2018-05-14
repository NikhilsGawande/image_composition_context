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
x_train_temp = []
y_train_temp = []
att_test_temp = []
non_att_test_temp = []
z = []
one = 1
minusone = -1


#Extracting Images list for atrributes 
att_list = read_data_list.read_att_list()
non_att_list = read_data_list.read_non_att_list()

counter = 0
for att in att_list :
	if att_list.index(att) < 1 :
		continue
	print("new attribute started")
	print(att_list.index(att))
	att_len = len(att)
	att_list_train = att[:int(0.7*att_len)]
	non_att_list_train = non_att_list[att_list.index(att)][:int(0.7*att_len)]
	att_list_test = att[int(0.7*att_len):]
	non_att_list_test = non_att_list[att_list.index(att)][int(0.7*att_len): att_len]


	for imagename in att_list_train :
		imgname = './../images/' + imagename	        	
		img = caffe.io.load_image(imgname)
		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = net.forward()	
		x_train_temp.append((net.blobs[layer].data[0].tolist())[:])
		y_train_temp.append(one)
		print(counter)
		counter += 1
	print("att_list")
	for imagename in non_att_list_train :		
		imgname = './../images/' + imagename
		img = caffe.io.load_image(imgname)
		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = net.forward()	
		x_train_temp.append((net.blobs[layer].data[0].tolist())[:])
		y_train_temp.append(minusone)
		print(counter)
		counter += 1
	print("non_att_list")
	for imagename in att_list_test :
		imgname = './../images/' + imagename
		img = caffe.io.load_image(imgname)
		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = net.forward()	
		att_test_temp.append((net.blobs[layer].data[0].tolist())[:])
		print(counter)
		counter += 1
	print("predicct_att_list")
	for imagename in non_att_list_test :
		imgname = './../images/' + imagename
		img = caffe.io.load_image(imgname)
		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = net.forward()	
		non_att_test_temp.append((net.blobs[layer].data[0].tolist())[:])
		print(counter)
		counter += 1
	print("predict_non_att_list")    
	print("features extracted")   
	x_train = np.array(x_train_temp)
	y_train = np.array(y_train_temp)

	x_test = np.array(att_test_temp)
	y_test = np.array(non_att_test_temp)


	feature_file = 'features_' + str(att_list.index(att)) + '.pkl'
	features = open(feature_file, 'wb')
	pickle.dump(x_train, features)
	features.close()

	label_file = 'labels_' + str(att_list.index(att)) + '.pkl'
	labels = open(label_file, 'wb')
	pickle.dump(y_train, labels) 
	labels.close()

	pp = 'predict_positive_' + str(att_list.index(att)) + '.pkl'
	predict_positive = open(pp, 'wb')
	pickle.dump(x_test, predict_positive)
	predict_positive.close()

	np = 'predict_negative_' + str(att_list.index(att))  + '.pkl'
	predict_negative = open(np, 'wb')
	pickle.dump(y_test, predict_negative)
	predict_negative.close()

	print("features stored")
	del(x_train_temp[:])
	del(y_train_temp[:])
	del(att_test_temp[:])
	del(non_att_test_temp[:])
	del(att_list_train[:])
	del(non_att_list_train[:])
	del(att_list_test[:]) 
	del(non_att_list_test[:])

