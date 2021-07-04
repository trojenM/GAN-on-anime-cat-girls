# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 01:53:28 2021

@author: troje
"""

import matplotlib.pyplot as plt
import torch
import cv2
import os
import numpy as np
from torch import nn
import torch.optim as opt
import torch.nn.functional as F
from customDataset import Catgirls
import math
import torchvision
import torchvision.transforms as transforms


device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


BATCH_SIZE = 128 
IMG_SIZE = 100

# Images to datas
image_datas = []

for image in os.listdir('./animecatgirls/'):
    try:
        img_list = cv2.imread(('./animecatgirls/' + image), cv2.IMREAD_GRAYSCALE)
        new_list = cv2.resize(img_list, (IMG_SIZE, IMG_SIZE))
        image_datas.append(new_list)
    except Exception as e:
        pass
    
new_image_datas = np.array(image_datas)
new_image_datas = np.resize(new_image_datas, (len(image_datas), IMG_SIZE, IMG_SIZE, 1))


train_data = Catgirls(new_image_datas, transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

data_iter = iter(train_loader)
image = data_iter.next()


# Generator class
Z = 100
H = 128
X = image.view(image.size(0), -1).size(1) 

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Z, H),
            nn.ReLU(),
            nn.Linear(H, X),
            nn.Sigmoid()
            )
        
    def forward(self, input):
        return self.model(input)
    
generator = Generator().to(device=device)

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X, H),
            nn.ReLU(),
            nn.Linear(H, 1),
            nn.Sigmoid()
            )
        
    def forward(self, input):
        return self.model(input)
        
discriminator = Discriminator().to(device=device)


def imshow(imgs):
    imgs = torchvision.utils.make_grid(imgs)
    npimgs = imgs.numpy()
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimgs, (1,2,0)), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


####
lr = 0.0002
g_opt = opt.Adam(generator.parameters(), lr)
d_opt = opt.Adam(discriminator.parameters(), lr)

for epoch in range(1000):
    G_loss_run = 0.0
    D_loss_run = 0.0
    
    for i, data, in enumerate(train_loader):
        x = data
        x = x.view(x.size(0), -1).to(device)
        BATCH_SIZE = x.size(0)
        
        one_labels = torch.ones(BATCH_SIZE, 1).to(device)
        zero_labels = torch.zeros(BATCH_SIZE, 1).to(device)
        
        z = torch.randn(BATCH_SIZE, Z).to(device)
        
        D_real = discriminator(x)
        D_fake = discriminator(generator(z))
        
        D_real_loss = F.binary_cross_entropy(D_real, one_labels)
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)
        D_loss = D_real_loss + D_fake_loss
        
        d_opt.zero_grad()
        D_loss.backward()
        d_opt.step()
        
        z = torch.randn(BATCH_SIZE, Z).to(device)
        D_fake = discriminator(generator(z))
        G_loss = F.binary_cross_entropy(D_fake, one_labels)
        
        g_opt.zero_grad()
        G_loss.backward()
        g_opt.step()
        
        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()
        
    print('Epoch:{},   G_loss:{},    D_loss:{}'.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1)))
    
    samples = generator(z).detach()
    samples = samples.view(samples.size(0), 1, 100, 100).cpu()
    imshow(samples)
    
            
        
