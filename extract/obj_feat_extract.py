import caffe
import numpy as np
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

obj_temp = []

#Extracting Images list for atrributes 
obj_list = read_data_list.read_obj_list()

counter = 0
for obj in obj_list :
	if(obj_list.index(obj) < 39 ) :
		continue
	for imagename in obj :		 
		imgname = './../images/' + imagename
		img = caffe.io.load_image(imgname)
		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = net.forward()	
		obj_temp.append((net.blobs[layer].data[0].tolist())[:])
		print(counter)
		counter += 1
	print("obj_list")

	print("features extracted")   
	x_train = np.array(obj_temp)


	feature_file = '../pkl/obj_features/features_' + str(obj_list.index(obj)) + '.pkl'
	features = open(feature_file, 'wb')
	pickle.dump(x_train, features)
	features.close()


	print("features stored")
	del(obj_temp[:])
	del(x_train)
	counter = 0
