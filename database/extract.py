# This code is to extract attribute and objects images list . 


import im_utils 
import _pickle as cPickle

image_data = im_utils.load('./mitstates_data/mit_image_data.pklz')
meta_data = im_utils.load('./mitstates_data/split_meta_info.pklz')

#Get list of images for attributes 
att_image_list = [[] for i in range (len(image_data['attributes']))]
for annotation in image_data['annotations'] :
	a = image_data['images'][annotation['image_id']]['file_name']
	b = annotation['att_labs']
	if(len(b) != 0 ) :
		#print(b[0])	
		att_image_list[b[0]].append(a)
	#print a

#Get list of images for objects
obj_image_list = [[] for i in range (len(image_data['objects']))]
for annotation in image_data['annotations'] :
	a = image_data['images'][annotation['image_id']]['file_name']
	b = annotation['ob_labs']
	if(len(b) != 0 ) :
		#print(b[0])	
		obj_image_list[b[0]].append(a)
	#print a

#Extracting examples for negative values of attributes. 
non_attr_final = []
non_attr = []
non = []
for attr in att_image_list :	
	for i in range(15) :
		for in_attr in att_image_list :
			if att_image_list.index(in_attr) != att_image_list.index(attr) :
				non.append(in_attr[i])
		for file in non :
			non_attr.append(file)
		del(non[:])
	non_attr_final.append(non_attr[:])
	del(non_attr[:])

#extracting examples for negative values of objects.
non_obj_final = []	
non_obj = []
for obj in obj_image_list :
	for i in range(2) :
		for in_obj in obj_image_list :
			if obj_image_list.index(in_obj) != obj_image_list.index(obj):
				non.append(in_obj[0])
		for file in non:
			non_obj.append(file)
		del(non[:])
	non_obj_final.append(non_obj[:])
	del(non_obj[:]) 

#extracting random pics for attributes :

random_att_final = []
random = []
for attr in att_image_list :
	for i in range(15) :
		random.append(attr[i])
	random_att_final.append(random[:])
	del(random[:])
	
#extracting random pics for objects :
random_obj_final = []
for obj in obj_image_list :
	for i in range(2) :
		random.append(obj[i])
	random_obj_final.append(random[:])
	del(random[:])
	
att_images = open('../pkl/att_images.pkl', 'wb')
cPickle.dump(att_image_list, att_images)
att_images.close()

obj_images = open('../pkl/obj_images.pkl', 'wb')
cPickle.dump(obj_image_list, obj_images)
obj_images.close()

non_att_images = open('../pkl/non_att_images.pkl', 'wb')
cPickle.dump(non_attr_final, non_att_images )
non_att_images.close()

non_obj_images = open('../pkl/non_obj_images.pkl', 'wb')
cPickle.dump( non_obj_final, non_obj_images)
non_obj_images.close()

random_att_images = open('../pkl/random_att_images.pkl', 'wb')
cPickle.dump(random_att_final, random_att_images)
random_att_images.close()


random_obj_images = open('../pkl/random_obj_images.pkl', 'wb')
cPickle.dump(random_obj_final, random_obj_images)
random_obj_images.close()
"""
>>> for x in image_data: 
...     print x
... 


image_data contains :
images
attributes
pairs
objects
annotations



meta_data contains :
>>> for x in meta_data:
...     print x 
... 
obIds
objects
pairNames
atIds
pkeys
attributes
pIds
pairSplitInfo

for attribute in image_data['attributes'] :
	for a in image_data['annotations']:
		if image_data['attributes'][a['att_labs']] == attribute
			print a['image_id'] 

 {u'att_labs': [114], u'image_id': 63439, u'pair_labs': [1961], u'ob_labs': [219]}


"""





