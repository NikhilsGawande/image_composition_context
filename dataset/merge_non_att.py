#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:48:03 2018

@author: nikmay
"""

import pickle


output_file = '../pkl/non_att_features/features_all.pkl'
temp_list = []
final_list = []
for i in range(115) :
	file = '../pkl/non_att_features/features_' + str(i) + '.pkl'
	pkl_file = open(file, 'rb')
	x_train = pickle.load(pkl_file)
	temp_list = x_train.tolist()
	final_list.append(temp_list[:])
	del(temp_list[:])


all_features = open('../pkl/non_att_features/features_all.pkl', 'wb')
pickle.dump(final_list, all_features)
all_features.close()
