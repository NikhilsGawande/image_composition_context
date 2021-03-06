import numpy as np
import tensorflow as tf
import math
import pickle
def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.leaky_relu(layer, alpha=0.1)

    return layer


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

#Get Input 
asvm = open('../pkl/network_train_att_svm.pkl', 'rb') 
att_svm = pickle.load(asvm)

osvm = open('../pkl/network_train_obj_svm.pkl', 'rb')
obj_svm = pickle.load(osvm)

feat = open('../pkl/network_train_features.pkl', 'rb')
features = pickle.load(feat)

yy = open('../pkl/network_train_y.pkl', 'rb')
yyy = pickle.load(yy)

#Input Classifiers
object_svm = tf.placeholder(tf.float32, [None, 1024], name='obj_svm')
attr_svm = tf.placeholder(tf.float32, [None, 1024], name='att_svm')
obj_attr = tf.concat([attr_svm, object_svm], 1, name='obj_attr')
y = tf.placeholder(tf.float32, [None], name='y_true')

#Transformation Network
"""
fc1 = tf.layers.dense(inputs=obj_attr, units= 3072) 
activation1 = tf.nn.leaky_relu(fc1, alpha=0.1)
fc2 = tf.layers.dense(inputs= activation1, units=1536)
activation2 = tf.nn.leaky_relu(fc2, alpha=0.1)
complex_features = tf.layers.dense(inputs = activation2, units=1024, activation=None)
activation3 = tf.nn.leaky_relu(complex_features, alpha = 0.1)
"""

fc1 = create_fc_layer(obj_attr, 2048, 3072, use_relu=True)
fc2 = create_fc_layer(fc1, 3072, 1536, use_relu=True)
complex_features = create_fc_layer(fc2, 1536, 1024, use_relu=False)
complex_features_n = tf.nn.l2_normalize(complex_features, axis=1, name='complex_feature') 
#1024 Dimensional feature vector of an image passed through VGG-M-1024
image_features = tf.placeholder(tf.float32, [None, 1024])

#Taking dot product of Image features and composed classifier
#dot_pro = tf.matmul(complex_features, tf.matrix_transpose(image_features))
#dot_pro  = tf.tensordot(complex_features, image_features, axes = 0)
dot_pro = tf.reduce_sum(tf.multiply(complex_features_n, image_features), 1)

pred = tf.nn.sigmoid(dot_pro, name='pred')
tf.Print(pred, [pred], "Hi")
#bin_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
#bin_cross_entropy= tf.keras.backend.binary_crossentropy(pred, y, from_logits = False)
bin_cross_entropy = -(tf.multiply(tf.transpose(y),log2(tf.clip_by_value(pred,1e-14, 1.0))) + tf.multiply(tf.transpose((1-y)),log2(tf.clip_by_value(1-pred, 1e-14, 1.0))))
#bin_cross_entropy = -(tf.multiply(y, log2(pred)) + tf.multiply((1-y) , log2(1 - pred)))
cost = tf.reduce_mean(bin_cross_entropy)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, momentum =0.9, decay=0.01).minimize(cost)
init = tf.global_variables_initializer()

"""
object_svm_b = 2 * np.random.random([25,1024]) -1
attr_svm_b = 2 * np.random.random([25,1024]) -1
image_features_b = 2 * np.random.random([25,1024]) -1
y_b = np.array([[0], [0], [0], [0], [0], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1]])
"""
#Initialise variables        
object_svm_b = obj_svm
attr_svm_b = att_svm
image_features_b = features
y_value = []
for i in yyy:
	y_value.append(i[0])
y_b = y_value
"""
y_array =  np.random.randint(1,2, size=9144)
y_b = []
y_temp=[]
for i in y_array:
    y_temp = []
    y_temp.append(i)
    y_b.append(y_temp)
"""

with tf.Session() as sess:
    sess.run(init)        
    # Training cycle
    training_epochs = 10
    for epoch in range(training_epochs):
        #for i in range(total_batch):
        # Run optimization op (backprop) and cost op (to get loss value)
        #total_batch = int(len(X)/batch_size)
        total_batch = 10
        #X_batches = np.array_split(X, total_batch)
        object_svm_batch = np.array_split(object_svm_b, total_batch)
        attr_svm_batch = np.array_split(attr_svm_b, total_batch)
        image_features_batch = np.array_split(image_features_b, total_batch)
        y_batch = np.array_split(y_b, total_batch)
        # Loop over all batches
        avg_cost = 0                    
        for i in range(total_batch):
            _, c, pr = sess.run([optimizer, cost,pred], feed_dict={object_svm:object_svm_batch[i], attr_svm:attr_svm_batch[i], image_features:image_features_batch[i], y:y_batch[i]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        #if epoch % display_step == 0:
            #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            #print("Optimization Finished!")
        print (avg_cost)
        saver = tf.train.Saver()
        
        #print(pr)
    saver.save(sess, './transform-network')
    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    #global result 
    #result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})


