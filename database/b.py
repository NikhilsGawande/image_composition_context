import im_utils 

a = im_utils.load('./mitstates_data/mit_image_data.pklz')




"""

There are total 63440 images 
attributes = 115
pairs = 1962


{'att_labs': [114], 'image_id': 63359, 'pair_labs': [1959], 'ob_labs': [121]}

images
attributes
pairs
objects
annotations
>>> a['attributes'][114]
'young'
>>> a['pairs'][1959]
[114, 121]
>>> a['objects'][121]
'horse'
>>> a['images'][0]
{'file_name': 'adj_aluminum/2005-Ford-Shelby-GR-1-Concept-Aluminum-RS-1280x960.jpg', 'image_id': 0}
>>> a['images'][0]['file_name']
'adj_aluminum/2005-Ford-Shelby-GR-1-Concept-Aluminum-RS-1280x960.jpg'
>>> a['images'][63359]['file_name']
'young_horse/young-horse-mother-standing-pasture-curiosity-alone-looking-me-33044939.jpg'
>>> 

"""
