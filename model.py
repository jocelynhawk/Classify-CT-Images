import numpy as np
import os
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Dense, Softmax
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from skimage.io import imread, imshow
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')),tf.config.experimental.list_physical_devices('GPU'))
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
            label = 0
        elif img_id[12:15] == 'PRO':
            label = 1
        else:
            print(img_id[12:15])
            continue
        label_TRAIN[i] = label
        i+=1
        
    return label_TRAIN, img_TRAIN

def AlexNet(img,label):
    inputs = Input((img_height,img_width,3))

    #normalize gs values
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)

    p6 = MaxPooling2D((2,2))(c5)
    p7 = MaxPooling2D((2,2))(p6)
    p8 = MaxPooling2D((2,2))(p7)


    n9 = BatchNormalization()(p7)
    n10 = BatchNormalization()(n9)

    fc11 = Dense(256,activation='relu')(n10)
    fc12 = Dense(256,activation='relu')(fc11)

    outputs = Softmax()(fc12)

    model = Model(inputs=[inputs],outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.summary()

    return model

#@jit(target_backend='cuda') 
def train(model):
    filepath = "model.h5"

    earlystopper = EarlyStopping(patience=10, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                save_best_only=True, mode='min')

    callbacks_list = [earlystopper, checkpoint]

    history = model.fit(img, label, validation_split=0.1, batch_size=16, epochs=100, 
                        callbacks=callbacks_list)



                
img_dir = '.\data'
label,img = load_images(img_dir)
model = AlexNet(img,label)
train(model)


#80,60 923,510
#input_layer = Input()
#c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')


