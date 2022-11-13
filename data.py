from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import cv2
import numpy as np

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, attgan_selected_attrs, stargan_selected_attrs):
        super(CelebA, self).__init__()
        self.data_path = data_path
        self.attr_path = attr_path
        self.selected_attrs = attgan_selected_attrs
        self.stargan_selected_attrs = stargan_selected_attrs
        # att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        # atts = [att_list.index(att) + 1 for att in attgan_selected_attrs]
        #this property images just imagename, not data
        # images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        # labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        images =  os.listdir(self.data_path)
        self.mode = mode
        
        if self.mode == 'train':
            self.images = images[:24000]
            # self.labels = labels[:182000]
        if self.mode == 'valid':
            self.images = images[24000:27000]
            # self.labels = labels[182000:182637]
        if self.mode == 'test':
            self.images = images[27000:]
            # self.labels = labels[182637:192637]
        
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)

        #stargan
        self.attr2idx = {}
        self.dataset = []
        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()

        #read attrname line
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
        
        #add stargan select lable with train/test/vaild data in dataset 
        #for example  [[000001.jpg, [True, True, False, True, False]], ....]
        for i, filename in enumerate(self.images):
            line_num = int(filename.split('.')[0]) + 2
            line = lines[line_num]
            values = line.split()[1:]
            if filename is not self.images[i]:
                continue
            label = []
            for attr_name in self.stargan_selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            self.dataset.append([filename, label])
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        # att = torch.tensor((self.labels[index] + 1) // 2)
        filename, label = self.dataset[index]
        return img, torch.FloatTensor(label)
        
    def __len__(self):
        return self.length



