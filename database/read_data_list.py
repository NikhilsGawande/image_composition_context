#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:26:33 2018

@author: nikmay
"""
#from _pickle import cPickle as pickle 
import pickle

#returns attribute list from att_images.pkl file
def read_att_list() :
    att_list = []
    pkl_file1 = open('../pkl/att_images.pkl', 'rb')
    att_list = pickle.load(pkl_file1)
    return att_list


#returns object list from obj_images.pkl file.
def read_obj_list() :
    obj_list = []
    pkl_file2 = open('../pkl/obj_images.pkl', 'rb')
    obj_list = pickle.load(pkl_file2)
    return obj_list

#returns attribute list from att_images.pkl file
def read_non_att_list() :
    non_att_list = []
    pkl_file3 = open('../pkl/non_att_images.pkl', 'rb')
    non_att_list = pickle.load(pkl_file3)
    return non_att_list 

#returns attribute list from att_images.pkl file
def read_non_obj_list() :
    non_obj_list = []
    pkl_file4 = open('../pkl/non_obj_images.pkl', 'rb')
    non_obj_list = pickle.load(pkl_file4)
    return non_obj_list 

def read_random_att() :
	random_list = []
	pkl_file5 = open('../pkl/random_att_images.pkl', 'rb')
	random_list = pickle.load(pkl_file5)
	return random_list

def read_random_obj() :
	random_list = []
	pkl_file6 = open('../pkl/random_obj_images.pkl', 'rb')
	random_list = pickle.load(pkl_file6)
	return random_list
	
																															
