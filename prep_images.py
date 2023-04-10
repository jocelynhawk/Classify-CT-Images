import os
import numpy as np
from skimage.io import imread, imsave
import shutil


def save_img(path,img_id,img):
    saved_imgs = os.listdir(path)
    for saved_id in saved_imgs:
        if saved_id == img_id:
            return
        
    imsave(path + img_id, img)

def main():         
    path = "data/raw/"
    images = os.listdir(path)
    for img_id in images:
        img = imread(path + img_id)
        img_crop = img[85:510,80:920]
        if img_id[12:15] == 'DIS':
            save_img('data/Distal/',img_id,img_crop)
        if img_id[12:15] == 'PRO':
            save_img('data/Proximal/',img_id,img_crop)

main()