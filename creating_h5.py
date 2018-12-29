'''
Creating h5py file
'''
from random import shuffle
import glob
import os

import numpy as np
import h5py

import cv2

__version__ ='1.0'
__update_date__ = '29-Dec-2018'

def creating_h5(train_path, test_path,
                pos_dir, neg_dir, pic_format='*.jpg',
                size_train=0.8, # split data into train and test set
                size=(64,64), # size of images
                ratio_pos_neg=2): # num of train images must be 2 times num of test images

    # INPUT: image paths
    # pos_path_format = pos_dir+'/'+pic_format
    # neg_path_format = neg_dir+'/'+pic_format
    #
    # pos_addrs = glob.glob(pos_path_format)
    # neg_addrs = glob.glob(neg_path_format)

    pos_addrs = [pos_dir+'/'+p for p in os.listdir(pos_dir)]
    neg_addrs = [neg_dir+'/'+n for n in os.listdir(neg_dir)]

    print('[INFO] Num pos images: ', len(pos_addrs))
    print('[INFO] Num neg images: ', len(neg_addrs))

    # SPLIT to train set and test set
    num_pos_train = int(len(pos_addrs)*size_train)
    num_neg_train = int(len(neg_addrs)*size_train)
    train_addrs = pos_addrs[:num_pos_train] + neg_addrs[:num_neg_train]
    test_addrs = pos_addrs[num_pos_train:] + neg_addrs[num_neg_train:]

    # label the data as 1 = car , 0 = noncar
    train_labels = [1]*len(pos_addrs[:num_pos_train]) + [0]*len(neg_addrs[:num_neg_train])
    test_labels = [1]*len(pos_addrs[num_pos_train:]) + [0]*len(neg_addrs[num_neg_train:])

    print('[INFO] Num train :', len(train_addrs))
    print('[INFO] Num test : ', len(test_addrs))

    ### Create h5 Object
    width = size[1]
    height = size[0]

    train_shape = (len(train_addrs), width , height)
    test_shape = (len(test_addrs), width , height)

    # open a hdf5 file and create arrays
    file_train = h5py.File(train_path, mode = 'w')
    file_test = h5py.File(test_path, mode = 'w')

    ## TRAIN
    ## file_train contains:
    ##      'list_classes': ['non-car', 'car']
    ##      'train_set_x': [image1,image2,...]
    ##      'train_set_y': [label1,label2,...]
    # create the "list_classes" for training dataset
    file_train.create_dataset("list_classes", (2,), 'S7')
    class_0 = 'non-car'
    class_1 = 'car'
    file_train["list_classes"][0] = class_0.encode("ascii", "ignore")
    file_train["list_classes"][1] = class_1.encode("ascii", "ignore")

    # create the "train_set_x" for training dataset
    file_train.create_dataset("train_set_x", train_shape, np.uint8)

    # create the "train_set_y" for training dataset
    file_train.create_dataset("train_set_y", (len(train_addrs), ), np.uint64)
    file_train["train_set_y"][...] = train_labels


    ## TEST
    ## file_test contains:
    ##      'list_classes': ['non-car', 'car']
    ##      'test_set_x': [image1,image2,...]
    ##      'test_set_y': [label1,label2,...]
    # create the "list_classes" testing dataset
    file_test.create_dataset("list_classes", (2,), 'S7')
    class_0 = 'non_car'
    class_1 = 'car'
    file_test["list_classes"][0] = class_0.encode("ascii", "ignore")
    file_test["list_classes"][1] = class_1.encode("ascii", "ignore")

    # create the "test_set_x" dataset
    file_test.create_dataset("test_set_x", test_shape, np.uint8)

    # create the "test_set_y" dataset
    file_test.create_dataset("test_set_y", (len(test_addrs), ), np.uint64)
    file_test["test_set_y"][...] = test_labels

    # loop over train paths
    for i,addr in enumerate(train_addrs):
        img = cv2.imread(addr)

        if img is None :
            print('[ERROR] Error reading img!', addr)
            continue

        if img[0,0].shape == (3,):
            # cv2 load images as BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize image to 64 x 64
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)

        file_train["train_set_x"][i, ...] = img[None] # same as img

    file_train.close()


    # loop over test paths
    for i,addr in enumerate(test_addrs):
        img = cv2.imread(addr)

        if img is None :
            print('[ERROR] Error reading img!', addr)
            continue

        if img[0,0].shape == (3,):
            # cv2 load images as BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize image to 64 x 64
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)

        file_test["test_set_x"][i, ...] = img[None]

    file_test.close()

    print('[INFO] Done !')

def show_info(train_path, test_path):
    file_train = h5py.File(train_path, mode = 'r')
    file_test = h5py.File(test_path, mode = 'r')
    print('[INFO] After checking ... ')
    print('[INFO] Num train:',file_train['train_set_x'].shape)
    print('[INFO] Num test:',file_test['test_set_x'].shape)

if __name__ == '__main__':
    print('''[INFO] File TRAIN contains:
         'list_classes': ['non-car', 'car']
         'train_set_x': [image1,image2,...]
         'train_set_y': [label1,label2,...]''')
    print('''[INFO] File TEST contains:
         'list_classes': ['non-car', 'car']
         'test_set_x': [image1,image2,...]
         'test_set_y': [label1,label2,...]''')
    print("[INFO] How to read: file_test = h5py.File(test_path, mode = 'r')")
    train_path = 'datasets/train_carvnoncar.h5'
    test_path = 'datasets/test_carvnoncar.h5'
    pos_dir = 'raw_datasets/pos'
    neg_dir = 'raw_datasets/neg'
    pic_format = '*.jpg'

    creating_h5(train_path, test_path, pos_dir, neg_dir, pic_format=pic_format, size_train = 0.8, size=(64,64), ratio_pos_neg=2)

    show_info(train_path, test_path)
