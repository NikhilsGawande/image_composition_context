"""
usage : 
# python3 run.py
This code uses VGG_CNN_M_1024.caffemodel which is trained on Imagenet database.  
Files needed : deploy.prototxt , out.npy, synset_words.txt 

""" 

import sys
import numpy as np
import caffe
import caffe.io

input_image_file = "input.jpeg" 
#output_file = "output_file"
model_file = "./VGG_CNN_M_1024.caffemodel"
deploy_prototxt = "./deploy.prototxt"

#initialize neural network
net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)

layer = 'fc7'
if layer not in net.blobs:
        raise TypeError("Invalid layer name: " + layer)


#We need to specify the image mean file for the image transformer:
imagemean_file = './out.npy'

#define transformer in order to preprocess the input image
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)

#reshape
net.blobs['data'].reshape(1,3,224,224)

img = caffe.io.load_image(input_image_file)

#run the image through preprocessor
net.blobs['data'].data[...] = transformer.preprocess('data', img)


output = net.forward()
"""
with open(output_file, 'w') as f:
        np.savetxt(f, net.blobs[layer].data[0], fmt='%.4f', delimiter='\n')

"""

#Predicting label
print (input_image_file, output['prob'].argmax())
label_mapping = np.loadtxt("synset_words.txt", str, delimiter='\t')
best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print (label_mapping[best_n])
print ("\n")




