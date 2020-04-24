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
        nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=pad),
        nn.BatchNorm2d(out, eps=1e-3),
        nn.ReLU(inplace=True)
    )

def weights_init_uniform(m):
    """
    Initializes the weights using kaiming method.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)
        
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)

class Branch(nn.Module):
    def __init__(self, bw):
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock3(1, bw, 2, 0),
                                     ConvBlock3(bw, bw, 1, 0),
                                     ConvBlock3(bw, bw, 1, 0),
                                     nn.MaxPool2d(2)
                                    )
        
        self.block2 = nn.Sequential(ConvBlock3(bw, bw*2, 1, 0),
                                    ConvBlock3(bw*2, bw*2, 1, 0),
                                    nn.MaxPool2d(2)
                                    )

        self.block3 = ConvBlock3(bw*2, bw*4, 1, 0)

    def forward(self, x):
        o1 = self.block1(x)
        o2 = self.block2(o1)
        o3 = self.block3(o2)
        return F.avg_pool2d(o3, o3.size()[2:]).view(x.size(0), -1)

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val

class OsteoSiameseNet(nn.Module):
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

class AdSiameseDataset(torch.utils.data.Dataset):
    """Some Information about OsteoSiameseDataset"""
    def __init__(self, pair_list, transform):
        super(AdSiameseDataset, self).__init__()

        self.transform = transform
        self.pair_list = pair_list

    def __getitem__(self, index):

        left, right = self.pair_list[index]
        try:
            label = int(left.split('/')[-2])
        except:
            label = int(np.random.randint(2))

        if left == right:
            if left.endswith('_0.png'):
                subject_name = left.split('/')[-1].replace('_0.png', '')
                left_, right_ = cv2.imread(left)[:, :, 0], cv2.imread(left)[:, :, 0]
        
            else:
                subject_name = left.split('/')[-1].replace('_1.png', '')
                right_ = np.fliplr(cv2.imread(right)[:, :, 0])
                left_, right_ = right_, right_

        else:
            subject_name = left.split('/')[-1].replace('_0.png', '')

            left_ = cv2.imread(left)[:, :, 0]
            right_ = np.fliplr(cv2.imread(right)[:, :, 0])
        
        left_ = self.transform(left_)
        right_ = self.transform(right_)

        data = dict(
            name = subject_name,
            image = [left_, right_],
            label = label
        )

        return data

    def __len__(self):
        return len(self.pair_list)

def train_test_split(pair_list, ratio=0.2):
    
    minors = [pair for pair in pair_list if pair[0].split('/')[-2] == str(1)]
    majors = [pair for pair in pair_list if pair[0].split('/')[-2] == str(0)]
    
    test_minor, train_minor = minors[:int(len(minors) * ratio)], minors[int(len(minors) * ratio):]
    test_major, train_major = majors[:int(len(majors) * ratio)], majors[int(len(majors) * ratio):]

    return test_minor+test_major, train_minor+train_major

def random_over_sampling(pair_list):
    minors = [pair for pair in pair_list if pair[0].split('/')[-2] == str(1)]
    majors = [pair for pair in pair_list if pair[0].split('/')[-2] == str(0)]

    sampled = []

    for i in range(len(majors)//len(minors)):

        sampled.extend(random.sample(minors, len(minors)))

    return majors + sampled

def getPairList(root_dir, duplicate = False):
    images = sorted(glob(os.path.join(root_dir, '*/*.png')))
    temp, pairs = dict(), dict()

    for img in images:
        subject = img.split('/')[-1].replace('.png', '')
        year, index, side = subject.split('_')

        try:
            temp['_'.join([year, index])].append(img)
            if len(temp['_'.join([year, index])]) == 2:
                pairs['_'.join([year, index])] = temp['_'.join([year, index])]

        except:
            temp['_'.join([year, index])] = [img]
            
    if duplicate:
        for key, value in temp.items():
            if len(value) == 1:
                pairs[key] = temp[key] * 2

    pair_list = list(pairs.values())
    return pair_list

def getExternalPairList(root_dir):
    images = sorted(glob(os.path.join(root_dir, '*.png')))
    temp, pairs = dict(), dict()

    for img in images:
        subject = img.split('/')[-1].replace('.png', '')
        index, side = subject.split('_')

        try:
            temp[index].append(img)
            if len(temp[index]) == 2:
                pairs[index] = temp[index]

        except:
            temp[index] = [img]

    pair_list = list(pairs.values())
    return pair_list

# Pair list 만들고 deep feature extraction 하는 코드 작성해보자 !!

data_dir = '/sdb1/share/ai_osteoporosis_brmh_png/dl/image/'
pair_list = getExternalPairList(data_dir)

if __name__ == "__main__":

    first_channels = 32
    BATCH_SIZE = 2
    num_classes = 5
    model = OsteoSiameseNet(bw=first_channels, drop=0.3, num_classes=num_classes)

    x1 = torch.Tensor(BATCH_SIZE, 1, 256, 256)
    x2 = torch.Tensor(BATCH_SIZE, 1, 256, 256)

    y = model(x1, x2)

    print(y.size())