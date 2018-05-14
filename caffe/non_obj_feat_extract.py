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

non_obj_temp = []
minusone = -1


#Extracting Images list for atrributes 
non_obj_list = read_data_list.read_random_obj()

counter = 0
for obj in non_obj_list :

	for imagename in obj :		
		imgname = './../images/' + imagename
		img = caffe.io.load_image(imgname)
		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = net.forward()	
		non_obj_temp.append((net.blobs[layer].data[0].tolist())[:])
		print(counter)
		counter += 1
	print("non_obj_list")

	print("features extracted")   
	x_train = np.array(non_obj_temp)


	feature_file = '../pkl/non_obj_features/features_' + str(non_obj_list.index(obj)) + '.pkl'
	features = open(feature_file, 'wb')
	pickle.dump(x_train, features)
	features.close()


	print("features stored")
	del(non_obj_temp[:])
	del(x_train)
	counter = 0