import tensorflow as tf
import pickle
import numpy as np
import im_utils
sess = tf.Session()
#recreating network graph. At this step only graph is created
saver = tf.train.import_meta_graph('transform-network.meta')

#Load the weights saved using the restore method
saver.restore(sess, tf.train.latest_checkpoint('./'))

#accessing the default graph which we have restored 
graph = tf.get_default_graph()

#output we want 
complex_feature = graph.get_tensor_by_name('complex_feature:0')

"""
asvm = open('../pkl/network_train_att_svm.pkl', 'rb') 
att_svms_temp = pickle.load(asvm)
att_svms_t = set(att_svms_temp)
att_svms = list(att_svms_t)

osvm = open('../pkl/network_train_obj_svm.pkl', 'rb')
obj_svms_temp = pickle.load(osvm)
obj_svms = list(set(obj_svms_temp))
"""
att_classifiers = open('../pkl/classifier_att_file.pkl', 'rb')
att_svmss = pickle.load(att_classifiers)

obj_classifiers = open('../pkl/classifier_obj_file.pkl', 'rb')
obj_svmss = pickle.load(obj_classifiers)

obj_svms = []
att_svms = []

image_data = im_utils.load('../pkl/mitstates_data/split_meta_info.pklz')
pair_keys = image_data['pkeys']

for pairs in pair_keys: 
	att_svms.append(att_svmss[pairs[0]])
	obj_svms.append(obj_svmss[pairs[1]])	

#all_possible_pairs = list(itertools.product(att_svms, obj_svms))

#feed the inputs 

obj_svm = graph.get_tensor_by_name('obj_svm:0')
att_svm = graph.get_tensor_by_name('att_svm:0')

total_batch = 5

#X_batches = np.array_split(X, total_batch)
object_svm_b = obj_svms
attr_svm_b = att_svms

object_svm_batch = np.array_split(object_svm_b, total_batch)
attr_svm_batch = np.array_split(attr_svm_b, total_batch)

cf_final = []
cf_temp = []
"""
for i in range(total_batch) :
	c_f = sess.run(complex_feature, feed_dict={obj_svm:object_svm_batch[i], att_svm:attr_svm_batch[i]})
	cf_temp = c_f.tolist()
	for i in cf_temp :
		cf_final.append(i)
"""
for i in range(len(pair_keys)):
	c_f = sess.run(complex_feature, feed_dict={obj_svm:object_svm_b[i], att_svm:attr_svm_b[i]})
	print(c_f)
	
	cf_temp = c_f.tolist()
	for i in cf_temp :
		cf_final.append(i)
	
print(len(c_f.tolist()))

c_features = open('../pkl/complex_features_forward_pass.pkl', 'wb')
pickle.dump(cf_final, c_features)





