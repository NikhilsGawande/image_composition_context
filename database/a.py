import im_utils 

m = im_utils.load('./mitstates_data/split_meta_info.pklz')
s_m_info = im_utils.load('./mitstates_data/split_meta_info.pklz')
i = im_utils.load('./mitstates_data/mit_image_data.pklz')

s_l_test =  im_utils.load('./mitstates_data/split_labels_test.pklz')
s_l_train = im_utils.load('./mitstates_data/split_labels_train.pklz')


ids = s_m_info 

'''
In split_meta_info.pkz

This things are present : 
obIds   	- Ids are given to objects 
objects		- Objects are given to names 
pairNames	- Pairnames are given (total - 1962)4
atIds		- Ids are given to attributes
attributes	- Attributes are given names 
pkeys		- Pair keys are given 
pIds		- Ids are given to pairs 
pairSplitInfo	- test train 1 and 0 are given 



objects : 245
known Pair combinations : 1962
No of attributes : 114



we have pairkeys and pairids 

'''