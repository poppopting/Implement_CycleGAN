#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import random


# In[19]:


class ImagePool():
    def __init__(self, size=50):
        self.size = size
        self.num_imgs = 0
        self.imgs = []
        
    def draw(self, images):
        return_imgs = []
        for img in images.data:
            img = img.unsqueeze(0)
            
            if self.num_imgs < self.size:
                self.imgs.append(img)
                return_imgs.append(img)
                self.num_imgs += 1
            else:
                p = random.random()
                
                if p > 0.5:
                    replace_id = random.randint(0, self.size-1)
                    return_img = self.imgs[replace_id].clone()
                    self.imgs[replace_id] = img
                    return_imgs.append(return_img)
                    
                else:
                    return_imgs.append(img)
        return_imgs = torch.cat(return_imgs, dim=0)
        return return_imgs

