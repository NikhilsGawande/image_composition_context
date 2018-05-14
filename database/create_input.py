#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 00:09:23 2018

@author: nikmay
"""

import pickle
from random import shuffle

a  = open('../pkl/pairs_data.pkl', 'rb')
p = pickle.load(a)

features_train = []
att_svms_train = []
obj_svms_train = []
features_train_pos = []
att_svms_train_pos = []
obj_svms_train_pos = []
features_train_neg = []
att_svms_train_neg = []
obj_svms_train_neg = []
y = []
one = [1]
zero = [0]
features_test = []
att_svms_test = []
obj_svms_test = []

for i in p : 
	images = []
	features = []
	att_svms = []
	obj_svms = []

	for j in i['images'] :
		images.append(j)
	for j in i['features'] :
		features.append(j.tolist())
	for j in i['obj_svms']: 
		obj_svms.append(j)
	for j in i['att_svms']: 
		att_svms.append(j)
	f_temp = []
	att_svms_temp = []
	obj_svms_temp = []
	f_pos_size = int(0.50*0.7*len(features))
	att_pos_size = int(0.50*0.7*len(att_svms))
	obj_pos_size = int(0.50*0.7*len(obj_svms))

	f_neg_size = int(0.7*len(features))
	att_neg_size = int(0.7*len(att_svms))
	obj_neg_size = int(0.7*len(obj_svms))

	f_temp = features[:f_pos_size]
	for j in f_temp :
		features_train_pos.append(j)
		y.append(one)	
	att_svms_temp = att_svms[:att_pos_size]
	for j in att_svms_temp :
		att_svms_train_pos.append(j)
	obj_svms_temp = obj_svms[:obj_pos_size]
	for j in obj_svms_temp :
		obj_svms_train_pos.append(j)

	f_temp = features[f_pos_size:f_neg_size]
	for j in f_temp :
		features_train_neg.append(j)		
	att_svms_temp = att_svms[att_pos_size:att_neg_size]
	for j in att_svms_temp :
		att_svms_train_neg.append(j)
	obj_svms_temp = obj_svms[obj_pos_size:obj_neg_size]
	for j in obj_svms_temp :
		obj_svms_train_neg.append(j)
	
	f_temp = []
	att_svms_temp = []
	obj_svms_temp = []
	f_temp = features[f_neg_size:]
	for j in f_temp :
		features_test.append(j)
	att_svms_temp = att_svms[att_neg_size:]
	for j in att_svms_temp :
		att_svms_test.append(j)
	obj_svms_temp = obj_svms[obj_neg_size:]
	for j in obj_svms_temp :
		obj_svms_test.append(j)


shuffle(features_train_neg)
shuffle(att_svms_train_neg)
shuffle(obj_svms_train_neg)
for i in range(len(features_train_neg)) :
	y.append(zero)

features_train = features_train_pos + features_train_neg
att_svms_train = att_svms_train_pos + att_svms_train_neg
obj_svms_train = obj_svms_train_pos + obj_svms_train_neg

d = []

for i in range(len(features_train)) :
	l = []
	l.append(i)
	l.append(features_train[i])
	l.append(att_svms_train[i])
	l.append(obj_svms_train[i])
	l.append(y[i])	
	d.append(l)
e = shuffle(d)
f_train = []
a_svms_train = []
o_svms_train = []
y_final = []
for i in range(len(features_train)) :
	f_train.append(d[i][1])
	a_svms_train.append(d[i][2])
	o_svms_train.append(d[i][3])
	y_final.append(d[i][4])

f = open('../pkl/network_train_features.pkl', 'wb')
pickle.dump(f_train, f)
f.close()



yy = open('../pkl/network_train_y.pkl', 'wb')
pickle.dump(y_final, yy)
yy.close()

asvm = open('../pkl/network_train_att_svm.pkl', 'wb')
pickle.dump(a_svms_train, asvm)
asvm.close()

osvm = open('../pkl/network_train_obj_svm.pkl', 'wb')
pickle.dump(o_svms_train, osvm)
osvm.close()

f = open('../pkl/network_test_features.pkl', 'wb')
pickle.dump(features_test, f)
f.close()

asvm = open('../pkl/network_test_att_svm.pkl', 'wb')
pickle.dump(att_svms_test, asvm)
asvm.close()

osvm = open('../pkl/network_test_obj_svm.pkl', 'wb')
pickle.dump(obj_svms_test, osvm)
osvm.close()


