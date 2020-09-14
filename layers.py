#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


# In[2]:


class c7s1_k(nn.Module):
    '''denote a 7×7 Convolution-InstanceNormReLU layer with k filters and stride 1'''
    def __init__(self, input_dim, k, norm=True, active='Relu'):
        super().__init__()
        self.active = {'Relu': nn.ReLU(True),
                       'Tanh': nn.Tanh()}
        
        self.models = nn.ModuleList([nn.ReflectionPad2d(3), 
                                     nn.Conv2d(in_channels=input_dim, out_channels=k, kernel_size=7, stride=1, bias=True)])
        if norm:
            self.models.append(nn.InstanceNorm2d(num_features=k))

        self.models.append(self.active[active])
        
        # nn.ReflectionPad2d(3)
    def forward(self, x):
        for layer in self.models:
            x = layer(x)
        return x


# In[3]:


class d_k(nn.Module):
    '''denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. '''
    def __init__(self, input_dim, k):
        super().__init__()
        
        self.models = nn.Sequential(nn.Conv2d(in_channels=input_dim, out_channels=k, kernel_size=3, stride=2, padding=1, bias=True),
                                    nn.InstanceNorm2d(num_features=k),
                                    nn.ReLU(True),
                                    )
        #nn.ReflectionPad2d(33)
    def forward(self, x):
        x = self.models(x)
        return x


# In[4]:


class R_k(nn.Module):
    '''denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer '''
    def __init__(self, input_dim, k):
        super().__init__()
        
        self.models = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels=input_dim, out_channels=k, kernel_size=3, stride=1, bias=True),
                                    nn.InstanceNorm2d(num_features=k),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, stride=1, bias=True),
                                    nn.InstanceNorm2d(num_features=k))
        #nn.ReflectionPad2d(33)
    def forward(self, x):
        x = x + self.models(x)
        return x


# In[5]:


class u_k(nn.Module):
    ''' denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2.'''
    def __init__(self, input_dim, k):
        super().__init__()
        
        self.models = nn.Sequential(nn.ConvTranspose2d(in_channels=input_dim, out_channels=k, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                                    nn.InstanceNorm2d(num_features=k),
                                    nn.ReLU(True),
                                    )
        #nn.ReflectionPad2d(33)
    def forward(self, x):
        x = self.models(x)
        return x


# In[6]:


class C_k(nn.Module):
    ''' denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2'''
    def __init__(self, input_dim, k, norm=True, stride=1):
        super().__init__()
        
        self.models = nn.ModuleList([nn.Conv2d(in_channels=input_dim, out_channels=k, kernel_size=4, stride=stride, padding=1)])
        if norm:
            self.models.append(nn.InstanceNorm2d(num_features=k))
        self.models.append(nn.LeakyReLU(0.2, True))
     
    def forward(self, x):
        for layer in self.models:
            x = layer(x)
        return x


# In[7]:


class Generator(nn.Module):
    '''c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3 '''
    
    def __init__(self, input_dim, n_blocks=6):
        super().__init__()
        self.models = nn.ModuleList([c7s1_k(input_dim=input_dim, k=64, active='Relu'),
                                     d_k(input_dim=64, k=128),
                                     d_k(input_dim=128, k=256)])
        for i in range(n_blocks):
            self.models.append(R_k(input_dim=256, k=256))

        self.models.extend([u_k(input_dim=256, k=128),
                            u_k(input_dim=128, k=64),
                            c7s1_k(input_dim=64, k=3, norm=False, active='Tanh')])
    
    def forward(self, x):
        for layer in self.models:
            x = layer(x)            
        return x


# In[13]:


class Discriminator(nn.Module):
    '''C64-C128-C256-C512'''
    
    def __init__(self, input_dim):
        super().__init__()
        self.models = nn.Sequential(C_k(input_dim=input_dim, k=64, norm=False, stride=2),
                                    C_k(input_dim=64, k=128, stride=2),
                                    C_k(input_dim=128, k=256, stride=2),
                                    C_k(input_dim=256, k=512, stride=1),
                                    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
                                   )
        
    def forward(self, x):
        x = self.models(x)
        return x

