import caffe
import numpy as np
import os 
#from sklearn import svm 
import pickle
import sys 
sys.path.insert(0, '../database/')
import read_data_list 

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

att = att_list[3]
att_len = len(att_list[3])
att_list_train = att_list[3][:int(0.7*att_len)]
non_att_list_train = non_att_list[3][:int(0.7*att_len)]
att_list_test = att_list[3][int(0.7*att_len):]
non_att_list_test = non_att_list[3][int(0.7*att_len) : att_len ]




"""
directory  = './elephant/elephants/'
for filename in os.listdir(directory):
	imgname = os.path.join(directory, filename)
"""
for imagename in att_list_train :
	imgname = './../images/' + imagename	        	
	img = caffe.io.load_image(imgname)
	net.blobs['data'].data[...] = transformer.preprocess('data', img)
	#print (imgname)
	#print (img)
	output = net.forward()	
	#a = np.array(output)
	#print (output)
	#print(net.blobs[layer].data[0])
	#z = net.blobs[layer].data[0].tolist()	
	#print(z)		
	x_train_temp.append((net.blobs[layer].data[0].tolist())[:])
	y_train_temp.append(one)
	#print(a)
	#del z[:]

#non elephants 
"""
directory  = './elephant/non_elephant/'
for filename in os.listdir(directory):
	imgname = os.path.join(directory, filename)
"""
for imagename in non_att_list_train :		
	imgname = './../images/' + imagename
	img = caffe.io.load_image(imgname)
	net.blobs['data'].data[...] = transformer.preprocess('data', img)
	#print (imgname)
	#print (img)
	output = net.forward()	
	#a = np.array(output)
	#print (output)
	#print(net.blobs[layer].data[0])
	#z = net.blobs[layer].data[0]
	#print(z)
	x_train_temp.append((net.blobs[layer].data[0].tolist())[:])
	y_train_temp.append(minusone)
	#print(a)


for imagename in att_list_test :
	imgname = './../images/' + imagename
	img = caffe.io.load_image(imgname)
	net.blobs['data'].data[...] = transformer.preprocess('data', img)
	#print (imgname)
	#print (img)
	output = net.forward()	
	#a = np.array(output)
	#print (output)
	#print(net.blobs[layer].data[0])
	#z = net.blobs[layer].data[0].tolist()	
	#print(z)		
	att_test_temp.append((net.blobs[layer].data[0].tolist())[:])

for imagename in non_att_list_test :
	imgname = './../images/' + imagename
	img = caffe.io.load_image(imgname)
	net.blobs['data'].data[...] = transformer.preprocess('data', img)
	#print (imgname)
	#print (img)
	output = net.forward()	
	#a = np.array(output)
	#print (output)
	#print(net.blobs[layer].data[0])
	#z = net.blobs[layer].data[0].tolist()	
	#print(z)		
	non_att_test_temp.append((net.blobs[layer].data[0].tolist())[:])
    
    
x_train = np.array(x_train_temp)
y_train = np.array(y_train_temp)

x_test = np.array(att_test_temp)
y_test = np.array(non_att_test_temp)


"""
feature_file = 'features.txt'
with open(feature_file, 'w+') as f:
	np.savetxt(f, x_train, fmt='%.4f', delimiter='\n')
"""
"""
label_file = 'labels.txt'
with open(label_file, 'w+') as f:
	np.savetxt(f, y_train, fmt='%.4f', delimiter='\n')
"""

features = open('features.pkl', 'wb')
pickle.dump(x_train, features)
features.close()

labels = open('labels.pkl', 'wb')
pickle.dump(y_train, labels) 
labels.close()

predict_positive = open('predict_positive.pkl', 'wb')
pickle.dump(x_test, predict_positive)
predict_positive.close()

predict_negative = open('predict_negative.pkl', 'wb')
pickle.dump(y_test, predict_negative)
predict_negative.close()


