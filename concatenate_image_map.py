import numpy as np
from scipy import misc
import glob

images_dir = '/work/isic2018-train-512/images/'
attr_dir = '/home/abissoto/isic2018-pytorch/isic-attr-512/'
output_dir = '/work/isic2018-train-512/concat/'

for image in glob.glob(images_dir + '*'):
    image_id = image.split('/')[-1].split('.')[0]
    #print(image_id)
    concat = misc.imread(image)
    #print(concat.shape)
    for attr in sorted(glob.glob(attr_dir + image_id + '*')):
        attr_load = misc.imread(attr)
        attr_load = np.expand_dims(attr_load, axis=-1)
        print(attr_load.shape)
        concat = np.concatenate((concat, attr_load), axis=-1)
    #print(concat.shape)
    np.save(output_dir+image_id+'.npy', concat)
