"""
Network architecture class
Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
modified by SangWook Kim, Seoul National University Hospital, 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle as pkl
import imageio
import numpy as np

from glob import glob
import os
import cv2
import random

def ConvBlock3(inp, out, stride, pad):
    """
    3x3 ConvNet building block with different activations support.
    
    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    return nn.Sequential(
        nn.Conv3d(inp, out, kernel_size=3, stride=stride, padding=pad),
        nn.BatchNorm3d(out, eps=1e-3),
        nn.ReLU(inplace=True)
    )

def weights_init_uniform(m):
    """
    Initializes the weights using kaiming method.
    """
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)
        
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)

class Branch(nn.Module):
    def __init__(self, bw):
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock3(1, bw, 1, 0),
                                     ConvBlock3(bw, bw, 1, 0),
                                     ConvBlock3(bw, bw, 1, 0),
                                     nn.MaxPool3d(2)
                                    )
        
        self.block2 = nn.Sequential(ConvBlock3(bw, bw*2, 1, 0),
                                    ConvBlock3(bw*2, bw*2, 1, 0),
                                    nn.MaxPool3d(2)
                                    )

        self.block3 = ConvBlock3(bw*2, bw*4, 1, 0)

    def forward(self, x):
        o1 = self.block1(x)
        o2 = self.block2(o1)
        o3 = self.block3(o2)
        return F.avg_pool3d(o3, o3.size()[2:]).view(x.size(0), -1)

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val

class AdSiameseNet(nn.Module):
    """
    Siamese Net to automatically grade osteoarthritis 
    
    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    def __init__(self, bw, drop, num_classes, use_w_init=True):
        super().__init__()
        self.branch = Branch(bw)

        if drop > 0:
            self.final = nn.Sequential(
                nn.Dropout(p=drop),
                nn.Linear(2*bw*4, num_classes),
                nn.Sigmoid()
            )

        else:
            self.final = nn.Sequential(
                nn.Linear(2*bw*4, num_classes),
                nn.Sigmoid()
            )
        
        # Custom weights initialization
        if use_w_init:
            self.apply(weights_init_uniform)

    def forward(self, x1, x2, save=False):
        """
        [ Feature-level fusion ]
        Here, Radiomics Feauture + deep features
        o1 -> left features, o2 -> right features

        Each radiomic features are already extracted from the images
        
        """
        # Shared weights
        o1 = self.branch(x1) # left
        o2 = self.branch(x2) # right

        if save:
            self.left_value = o1
            self.right_value = o2
        feats = torch.cat([o1, o2], 1)
        return self.final(feats)

def getPairList(left_dir, right_dir):
    left_list = sorted(glob(os.path.join(left_dir, '*.npy')))
    right_list = sorted(glob(os.path.join(right_dir, '*.npy')))
    pair_list = list(zip(left_list, right_list))
    return pair_list

def random_over_sampling(majors, minors):
    sampled = []
    for i in range(len(majors)//len(minors)):
        sampled.extend(random.sample(minors, len(minors)))
    return majors + sampled

def train_test_split(pair_list, train_ratio=0.8, ros=True):

    label_major, label_minor = [], []

    with open('../Data/ad_radiomics/label_dict.pickle', 'rb') as f:
        labels = pkl.load(f)

    for pair in pair_list:
        subject = pair[0].split('/')[-1].replace('.npy', '')

        label = labels[subject]

        if label in [0, 1]:
            label_major.append(pair)

        else:
            label_minor.append(pair)

    major_train, major_test = label_major[:int(len(label_major) * train_ratio)], label_major[int(len(label_major) * train_ratio):]
    minor_train, minor_test = label_minor[:int(len(label_minor) * train_ratio)], label_minor[int(len(label_minor) * train_ratio):]
    
    if ros:
        train_pairs = random_over_sampling(major_train, minor_train)
    else:
        train_pairs = major_train + minor_train
        
    test_pairs = major_test + minor_test
    return train_pairs, test_pairs

class AdSiameseDataset(torch.utils.data.Dataset):
    """Some Information about AdSiameseDataset"""
    def __init__(self, pair_list, transform):
        super(AdSiameseDataset, self).__init__()
        
        self.pair_list = pair_list
        self.transform = transform
        
        with open('../Data/ad_radiomics/label_dict.pickle', 'rb') as f:
            self.labels = pkl.load(f)

    def __getitem__(self, index):

        left, right = self.pair_list[index]
        label = int(self.labels[left.split('/')[-1].replace('.npy', '')])
        label = 0 if label in [0, 1] else 1

        subject_name = left.split('/')[-1].replace('.npy', '')
        
        left_ = np.load(left)
        right_ = np.flipud(np.load(right))
        
        left_ = self.transform(left_).unsqueeze(0)
        right_ = self.transform(right_).unsqueeze(0)

        data = dict(
            name = subject_name,
            image = [left_, right_],
            label = label
        )
        return data

    def __len__(self):
        return len(self.pair_list)

if __name__ == "__main__":

    first_channels = 32
    BATCH_SIZE = 4
    num_classes = 2
    model = AdSiameseNet(bw=first_channels, drop=0.3, num_classes=num_classes)

    x1 = torch.Tensor(BATCH_SIZE, 1, 32, 32, 32)
    x2 = torch.Tensor(BATCH_SIZE, 1, 32, 32, 32)

    y = model(x1, x2)
    print(y.size())