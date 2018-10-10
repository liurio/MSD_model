from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import os
import glob
from time import gmtime,strftime

def data_files():
    has_glass_files = glob.glob('/home/eric/CelebA/Img/img_celeba.7z/hasGlassAlign/*.jpg')
    no_glass_files = glob.glob('/home/eric/CelebA/Img/img_celeba.7z/noGlassAlign/*.jpg')
    np.random.shuffle(has_glass_files)
    np.random.shuffle(no_glass_files)
    m = len(has_glass_files)
    n = len(no_glass_files)
    print(m,n,0.05*m)
    has_glass_files_train = has_glass_files[:int(m)]
    has_glass_files_val = has_glass_files[int(0.95*m):]
    no_glass_files = np.random.choice(no_glass_files,m)
    no_glass_files_train = no_glass_files[:int(0.95*m)]
    no_glass_files_val = no_glass_files[int(0.95*m):]
    print('trainA:%d,trainB:%d,valA:%d,valB:%d'%(len(has_glass_files_train),len(no_glass_files_train,
                                                                                len(has_glass_files_val),len(no_glass_files_val))))
    for i in range(len(has_glass_files_val)):
        f=open('val_imgfiles.txt','a')
        f.write(has_glass_files_val[i]+'/'+no_glass_files_val[i]+'/n')

    return no_glass_files_train, has_glass_files_train,no_glass_files_val, has_glass_files_val

def load_data(image_path, flip=True, is_test=False, image_size = 150):
    img = load_image(image_path)
    img = preprocess_img(img, img_size=image_size, flip=flip, is_test=is_test)
    if is_test==False:
        index = np.random.randint(0,22,2)
        img=img[index[0]:index[0]+128,index[1]:index[1]+128]

    img = img/127.5 - 1.
    if len(img.shape)<3:
        img = np.expand_dims(img, axis=2)
    return img

def load_image(image_path):
    img = imread(image_path)
    return img

def preprocess_img(img, img_size=128, flip=True, is_test=False):
    img = scipy.misc.imresize(img, [img_size, img_size])
    if (not is_test) and flip and np.random.random() > 0.5:
        img = np.fliplr(img)
    return img

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    dir = os.path.dirname(image_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True)#.astype(np.float)
    else:
        return scipy.misc.imread(path)#.astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if len(images.shape) < 4:
        img = np.zeros((h * size[0], w * size[1], 1))
        images = np.expand_dims(images, axis = 3)
    else:
        img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    if images.shape[3] ==1:
        return np.concatenate([img,img,img],axis=2)
    else:
        return img.astype(np.uint8)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return ((images+1.)*127.5)
