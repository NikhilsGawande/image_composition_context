





import sys
sys.path.insert(0, '../database/')
import im_utils
import numpy as np
import tensorflow as tf
import pickle 

#Input Classifiers
att_classifier = open('../pkl/classifier_file.pkl', 'rb') 
obj_classifier = open('../pkl/classifier_obj_file.pkl',  'rb') 
obj_class = pickle.load(obj_classifier)
att_class = pickle.load(att_classifier)

#Input features 
features = open('../pkl/pairs_data.pkl', 'rb')
images = im_utils.load('./mitstates_data/mit_image_data.pklz')
feat_list = pickle.load(features)


object_svm = tf.placeholder(tf.float32, [None, 1024])
attr_svm = tf.placeholder(tf.float32, [None, 1024])
obj_attr = tf.concat([attr_svm, object_svm], 0)

#Transformation Network
fc1 = tf.layers.dense(inputs=obj_attr, units= 3072, activation=tf.nn.leaky_relu)
fc2 = tf.layers.dense(inputs= fc1, units=1536, activation=tf.nn.leaky_relu)
complex_features = tf.layers.dense(inputs=fc2, units=1024, activation=None)


pair_key = images['pairs'][0]
pair_count = len()



#1024 Dimensional feature vector of an image passed through VGG-M-1024
image_features = tf.placeholder(tf.float32, [None, 1024])

learning_rate = 0.001
#Taking dot product of Image features and composed classifier
dot_pro = tf.matmul(complex_features, tf.matrix_transpose(image_features))
pred = tf.sigmoid(dot_pro)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

		
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    global result 
    result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})

















