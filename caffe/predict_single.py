import pickle 
import tensorflow as tf
import numpy as np 
import im_utils 

a = im_utils.load('../pkl/mitstates_data/split_meta_info.pklz')

pairs = a['pairNames']
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

feat = open('../pkl/image_feature.pkl', 'rb')
feature = pickle.load(feat)


#feat = test_data[1]['features'][0]
feat = feature

dot_pro = tf.reduce_sum(tf.multiply(tf_cf, tf_feat))
sess = tf.Session()


score = []
for cf in complex_features :
	dot_pr = sess.run(dot_pro, feed_dict= {tf_cf:cf , tf_feat:feat}) 
	score.append(dot_pr)
top_five = sorted(range(len(score)), key=lambda i: score[i])[-5:]
print('Output')
for i in top_five :
	print(pairs[i], end=' ') 
print()
#print(features['pIds'][i])
