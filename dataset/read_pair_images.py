# -*- coding: utf-8 -*-
import im_utils
import sys
sys.path.insert(0, './../dataset/')
import read_data_list 
import pickle 
images = im_utils.load('../pkl/mitstates_data/mit_image_data.pklz')
split_info = a = im_utils.load('../pkl/mitstates_data/split_meta_info.pklz')

att_images = read_data_list.read_att_list()
pairs_train_data = []
pairs_test_data = []
a = open('../pkl/classifier_file.pkl', 'rb')
att_svms = pickle.load(a)

a = open('../pkl/classifier_obj_file.pkl', 'rb')
obj_svms = pickle.load(a)

no_pairs = len(images['pairs'])

pair_split_info = split_info['pairSplitInfo']
psi_train = pair_split_info['train']
for i in range(340) :
	pair_key = images['pairs'][i]
	
	pair_name = images['attributes'][pair_key[0]] + '_' + images['objects'][pair_key[1]]
	print(pair_name)
	pair_images = []
	pair_features = []
	att_svm = []
	obj_svm = []
	pids = []
	pair_dict = {}
	feature_att_file = '../pkl/att_features/features_' + str(pair_key[0]) + '.pkl'
	att_file = open(feature_att_file, 'rb')
	att_features = pickle.load(att_file)
	att = att_images[pair_key[0]]
	for j in att :
		if pair_name in j:
			print(images['attributes'][pair_key[0]] + '_' + images['objects'][pair_key[1]]) 
			pair_images.append(j)
			pair_features.append(att_features[att.index(j)])
			att_svm.append(att_svms[pair_key[0]][0])
			obj_svm.append(obj_svms[pair_key[1]][0])
			pids.append(i)	
	pair_dict['att_name'] = images['attributes'][pair_key[0]]
	pair_dict['att_id'] = pair_key[0]
	pair_dict['obj_name'] = images['objects'][pair_key[1]]
	pair_dict['obj_id'] = pair_key[1]
	pair_dict['images'] = pair_images
	pair_dict['features'] = pair_features
	pair_dict['att_svms'] = att_svm
	pair_dict['obj_svms'] = obj_svm
	pair_dict['pId'] = pids
	if psi_train[i] == 1 :
		pairs_train_data.append(pair_dict)
	else :
		pairs_test_data.append(pair_dict)		
#	del(pair_images[:])
#	del(pair_features[:])
	
#	del(att[:])
	print(i)

result_file = open('../pkl/pairs_train_data.pkl', 'wb')
pickle.dump(pairs_train_data, result_file)
result_file.close()

result_file = open('../pkl/pairs_test_data.pkl', 'wb')
pickle.dump(pairs_test_data, result_file)
result_file.close()


