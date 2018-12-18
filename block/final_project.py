import person_blocker
import os
import argparse
#from classes import get_class_names, InferenceConfig
from brands import get_class_names, InferenceConfig
import imageio



parser = argparse.ArgumentParser(
    description = 'Person Blocker - Automatically "block" people in images using a neural network.'
)
parser.add_argument('-i', '--image', help='Image file name', required=False)
parser.add_argument('-gpu', '--GPU', help='The number of GPU', type=str)
#parser.add_argument('-of', '--outfile', help='Output file name', required=False)
'''
parser.add_argument('-m', '--model', help='path to COCO model', default=None)
parser.add_argument('-o', '--objects', nargs='+', 
                    help='object(s)/object ID(s) to block. ' +
                    'Use the -names flag to print a list of ' +
                    'valid objects', default='person')
parser.add_argument('-c', '--color', nargs='?', default='(255, 255, 255)')
parser.add_argument('-l', '--labeled', dest='labeled', action='store_true',
                    help='generate labeled image instead')
parser.add_argument('-n', '--name', dest='names', action='store_true', 
                    help='prints class names and exits')
# Not the same as the original one
parser.set_defaults(labeled=True, names=False)
'''
args = parser.parse_args()

# Only use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU



class Argums():
    def __init__(self, image, model, objects, color, labeled, name):
        self.image = image
        #self.model = model
        self.model = 'mask_rcnn_brands.h5'
        self.objects = objects
        self.color = color
        self.labeled = labeled
        self.name = name


# Label file
#outfile1 = 'labeled.png'
#args1 = Argums(args.image, None, 'person', '(255, 255, 255)', True, False)
#result, position_ids = person_blocker.person_blocker(args1, None)
#print('########################################')
#print(result)
#print('########################################')
#print(position_ids)
#print('########################################')

# Split the pictures
#image = imageio.imread(args.image)
#boxes = result['rois']
#class_ids = result['class_ids']

#N  = boxes.shape[0]
'''
for i in range(N):
    y1, x1, y2, x2 = boxes[i]
    #split_img = image[x1:x2, y1:y2]
    split_img = image[y1:y2, x1:x2]
    imageio.imwrite('split%d.png'%i, split_img)
'''


# Output file
print('########################################')
print('start')
outfile2 = 'eraser.png'
args2 = Argums(args.image, None, 'rittersport', '(255, 255, 255)', False, False)
person_blocker.person_blocker(args2, outfile2)
print('########################################')
print('end')




print('finished')
