import numpy as np
import os
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Dense, Softmax, Flatten
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import optimizers
from skimage.io import imread, imshow
from skimage.transform import resize
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')),tf.config.experimental.list_physical_devices('GPU'))
img_width, img_height = 256, 128


def load_images(folder_path):
    images_dis = os.listdir(folder_path + 'Distal') 
    images_pro = os.listdir(folder_path + 'Proximal')
    images_train = os.listdir(folder_path + 'Train')
    images_test = os.listdir(folder_path + 'Test')
    img_TRAIN = np.zeros((len(images_train),img_height,img_width),dtype=np.uint8)
    img_TEST = np.zeros((len(images_test),img_height,img_width),dtype=np.uint8)
    label_TRAIN = np.zeros((len(images_train),2))
    i=0
    for img_id in images_train:
        img = imread('data/Train/' + img_id,as_gray=True)
        img=(img[:,:]-128)/128
        img = resize(img,(128,256))
        img_TRAIN[i] = img
        if img_id[12:15] == 'PRO':
            label_TRAIN[i] = np.array([1,0])
        elif img_id[12:15] == 'DIS':
            label_TRAIN[i] = np.array([0,1])
        else:
            print('ERROR: ',img_id)
        i+=1
    
    """for img_id in images_dis:
        img = imread('data/Distal/' + img_id,as_gray=True)
        img = resize(img,(128,256))
        imshow(img)
        img_TRAIN[i] = img
        label_TRAIN[i] = np.array([1,0])
        i+=1"""



    i=0
    for img_id in images_test:
        img = imread('data/Test/' + img_id,as_gray=True)
        img = resize(img,(128,256))
        img_TEST[i] = img
        i+=1
        
    return label_TRAIN, img_TRAIN, img_TEST

def AlexNet():
    model = tf.keras.models.Sequential([ 
        #1st Convolutional Layer
        tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation='relu', input_shape=(128,256,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        #2nd Convolutional Layer
        tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        #3rd Convolutional Layer
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        #4th Convolutional Layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        #5th Convolutional Layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        #Passing it to a Fully Connected layer
        tf.keras.layers.Flatten(),
        # 1st Fully Connected Layer
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),# Add Dropout to prevent overfitting
        # 2nd Fully Connected Layer
        tf.keras.layers.Dense(4096, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.5),
        # 3rd Fully Connected Layer
        tf.keras.layers.Dense(1000, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.5),
        #Output Layer
        tf.keras.layers.Dense(2, activation='softmax'),
        #tf.keras.layers.BatchNormalization()
    ])

    model.compile(
        optimizer=tf.optimizers.SGD(learning_rate=0.00000001),
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )

    model.summary()

    return model

#@jit(target_backend='cuda') 
def train(model,img,label):
    filepath = "model.h5"

    earlystopper = EarlyStopping(patience=10, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                save_best_only=True, mode='min')

    callbacks_list = [earlystopper, checkpoint]

    history = model.fit(img, label, validation_split=0.1, batch_size=25, epochs=100, 
                        callbacks=callbacks_list)
    #output = model(**batch)

    
    



def main():               
    label,img, test = load_images('data/')
    print("CHECK LENGTHS: ",len(label),len(img))
    model = AlexNet()
    print('BEGINNNING TRAINING')
    train(model,img,label)
    
    model.load_weights('model.h5')
    test_preds = model.predict(img[5:15])
    
    print(test_preds)
    print(label[5:15])

main()


#80,60 923,510
#input_layer = Input()
#c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')


