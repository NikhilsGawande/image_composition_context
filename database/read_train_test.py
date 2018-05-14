#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:18:59 2018

@author: nikmay
"""
# -*- coding: utf-8 -*-
import pickle

train_file = open('./train_data.pkl', 'rb')
a = pickle.load(train_file, encoding='latin1')


with open('./train_data.pkl', 'rb') as f:
	u = pickle._Unpickler(f)
	u.encoding = 'latin1'
	p = u.load()
	