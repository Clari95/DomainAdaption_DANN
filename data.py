# -*- coding: utf-8 -*-
"""data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kjy0kzD5EzGB7nZBc_GIMkxy2aRgCd2L
"""

import os
from os import listdir
import json
import torch
import scipy.misc
import numpy 
import glob
import csv

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class Mnist(Dataset):
    def __init__(self, args, mode ='train', visualization=False):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, 'digits/mnistm/' + mode)
        if visualization:
                print('viz_data')
                self.img_dir = os.path.join(self.data_dir, 'digits/mnistm/test')
        self.img_dir_csv = os.path.join(self.data_dir, 'digits/mnistm/' + mode + '.csv')
        if visualization:
                self.img_dir_csv = os.path.join(self.data_dir, 'digits/mnistm/test.csv')
        self.img_paths = []
        self.img_labels = []
        print(self.img_dir)
        print(self.img_dir_csv)
       
        ''' read the data list '''
        
        csv_path = self.img_dir_csv

        f = open(csv_path, 'r')
        data_list = f.readlines()
        del data_list[0]
        f.close()
        self.n_data = len(data_list)

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])
     

        
        
        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                              # transforms.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])


    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        ''' get data '''
       
        img_paths, labels = self.img_paths[idx], self.img_labels[idx]
        ''' read image '''
        img = Image.open(os.path.join(self.img_dir, img_paths)).convert('RGB')
        imgs = self.transform(img)
        labels = int(labels)
       
        return imgs , labels


class SVHN(Dataset):
    def __init__(self, args, mode ='train', visualization=False):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        
        self.img_dir = os.path.join(self.data_dir, 'digits/svhn/' + mode)
        if visualization:
                self.img_dir = os.path.join(self.data_dir, 'digits/svhn/test')
        self.img_dir_csv = os.path.join(self.data_dir, 'digits/svhn/' + mode + '.csv')
        if visualization:
                self.img_dir_csv = os.path.join(self.data_dir, 'digits/svhn/test.csv')
        self.img_paths = []
        self.img_labels = []
       
        ''' read the data list '''
        
        csv_path = self.img_dir_csv

        f = open(csv_path, 'r')
        data_list = f.readlines()
        del data_list[0]
        f.close()
        self.n_data = len(data_list)

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])
     

        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                              # transforms.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])


    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        ''' get data '''
       
        img_paths, labels = self.img_paths[idx], self.img_labels[idx]
        ''' read image '''
        img = Image.open(os.path.join(self.img_dir, img_paths)).convert('RGB')
  
       
        return self.transform(img), int(labels)




class DATA_TEST(Dataset):
    def __init__(self, args, mode ='test'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir)
        self.img_dir_csv = os.path.join(self.data_dir + '.csv')
        self.img_paths = []
        self.img_labels = []
       
        ''' read the data list '''
        
        csv_path = self.img_dir_csv

        f = open(csv_path, 'r')
        data_list = f.readlines()
        del data_list[0]
        f.close()
        self.n_data = len(data_list)

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])
     

        
        
        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                              # transforms.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])


    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        ''' get data '''
       
        img_paths, labels = self.img_paths[idx], self.img_labels[idx]
        ''' read image '''
        img = Image.open(os.path.join(self.img_dir, img_paths)).convert('RGB')
        imgs = self.transform(img)
        labels = int(labels)
       
        return imgs , labels

class face(Dataset):
    def __init__(self, args, mode ='train', visualization=False):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        print('get datadir')
        self.img_dir = os.path.join(self.data_dir, 'face/' + mode)
        self.img_dir_csv = os.path.join(self.data_dir, 'face/' + mode + '.csv')
        self.img_paths = []
        self.img_labels = []
        image_size = 64
       
        ''' read the data list '''
        
        csv_path = self.img_dir_csv

        f = open(csv_path, 'r')
        data_list = f.readlines()
        del data_list[0]
        f.close()
        self.n_data = len(data_list)

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])
     

        ''' set up image trainsform '''
        if self.mode == 'train':
            self.dataset = dset.ImageFolder(root=args.data_dir,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
            #self.transform = transforms.Compose([
            #                  # transforms.RandomHorizontalFlip(0.5),
            #                   transforms.Resize(image_size),
            #                   transforms.CenterCrop(image_size),
            #                   transforms.ToTensor(),
            #                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               ransforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                               ])


    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        ''' get data '''
       
       # img_paths, labels = self.img_paths[idx], self.img_labels[idx]
        ''' read image '''
       # img = Image.open(os.path.join(self.img_dir, img_paths)).convert('RGB')
       
       
        return self.dataset

