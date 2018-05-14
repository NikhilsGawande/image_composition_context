import pickle 
import tensorflow as tf
import numpy as np 

tf_cf = tf.placeholder(tf.float32, [1024], name='complex_featur')
tf_feat = tf.placeholder(tf.float32, [1024], name='feature')


c_features = open('../pkl/complex_features_forward_pass.pkl', 'rb')
complex_features = pickle.load(c_features)
for i in range(len(complex_features)) :
	complex_features[i] = np.asarray(complex_features[i])
'''
test = open('../pkl/pairs_test_data.pkl', 'rb')
test_data = pickle.load(test)
'''

feat = open('../pkl/network_test_features.pkl', 'rb')
features = pickle.load(feat)


#feat = test_data[1]['features'][0]
feat = features['features'][0]

dot_pro = tf.reduce_sum(tf.multiply(tf_cf, tf_feat))
sess = tf.Session()


for i in range(len(features['features'])) :	
	score = []
	for cf in complex_features :
		feat = features['features'][i]
		dot_pr = sess.run(dot_pro, feed_dict= {tf_cf:cf , tf_feat:feat}) 
		score.append(dot_pr)
	print(sorted(range(len(score)), key=lambda i: score[i])[-5:])
	print(features['pIds'][i])
