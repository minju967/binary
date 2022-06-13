from PIL import Image
from torchvision import transforms

import numpy as np
import torch.utils.data
import os
import torch
import pandas as pd
import random

class SingleviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, train_list, f_img_path, feature):
        self.csv_path = train_list
        self.feature = feature
        self.raw_data = pd.read_csv(self.csv_path, names=['img_path', 'class'])

        self.filepaths = []
        for idx in range(len(self.raw_data)):
            self.filepaths.append(self.raw_data.iloc[idx, 0])
        self.feature_list = os.listdir(os.path.join(f_img_path, feature))

        random.shuffle(self.filepaths)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        file_name = path.split('/')[-1]
        if file_name in self.feature_list:
            class_id = 1
        else:
            class_id = 0

        tf = transforms.ToTensor()
        im = Image.open(self.filepaths[idx]).convert('RGB')

        if self.transform:
            im = tf(im)
            im = self.transform(im)

        return (class_id, im, path)


class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, train_list, nviews, shuffle=False):
        self.num_views = nviews
        self.csv_path  = train_list
        self.raw_data  = pd.read_csv(self.csv_path, names=['img_path', 'class'])
        self.classnames = ['A', 'B', 'C', 'D', 'E']
        self.mean = [0.46147138, 0.45751584, 0.44702336]
        self.std = [0.20240466, 0.19746633, 0.18430763]
        self.filepaths = []
        for idx in range(len(self.raw_data)):
            self.filepaths.append(self.raw_data.iloc[idx, 0])

        self.filepaths = sorted(self.filepaths)
        if shuffle == True:
            rand_idx = np.random.permutation(int(len(self.filepaths)/self.num_views))
            new_dataset = []

            for i in range(len(rand_idx)):
                new_dataset.extend(self.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])

            self.filepaths = new_dataset

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return int(len(self.filepaths)//self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-2]
        class_id   = self.classnames.index(class_name)

        imgs = []
        tf = transforms.ToTensor()
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = tf(im)
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views], path.split('/')[-1], class_name)