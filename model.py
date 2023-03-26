import numpy as np
import keras
import os
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from skimage.io import imread, imshow

img_width, img_height = 840, 450


def load_images(folder_path):
    images = os.listdir(folder_path)
    img_TRAIN = np.zeros((len(images),img_height,img_width,3),dtype=np.uint8)
    label_TRAIN = np.zeros(len(images))
    i=0
    for img_id in images:
        img = imread('.\data\\' + img_id)
        img_TRAIN[i] = img[60:510,80:920]
        if img_id[12:15] == 'DIS':
            label = 2
        elif img_id[12:15] == 'PRO':
            label = 1
        else:
            print(img_id[12:15])
            continue
        label_TRAIN[i] = label
        i+=1
        
    print(label_TRAIN)

                
img_dir = '.\data'
load_images(img_dir)

#80,60 923,510
#input_layer = Input()
#c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')


