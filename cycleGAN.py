# basic package
import os
import gc
import glob
import random
import numpy as np
import itertools
from PIL import Image
from tqdm.auto import tqdm
# pytorch package
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# my package
from utils import ImagePool
from layers import Generator, Discriminator


root = '/home/andy_cgt/summer2winter_yosemite'

# create dataset
class ImageFolder(Dataset):
    
    def __init__(self, root_dir, mode, transform=None):
        self.files_A = glob.glob(os.path.join(root_dir, '{0}A/*.jpg'.format(mode)))
        self.files_B = glob.glob(os.path.join(root_dir, '{0}B/*.jpg'.format(mode)))
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)
        self.transform = transform
    
    def __getitem__(self, index):
        imgA = Image.open(self.files_A[index % self.len_A])
        # avoid fixed pairs
        index_B = random.randint(0, self.len_B - 1)
        imgB = Image.open(self.files_B[index_B])

        if self.transform is not None:
            return self.transform(imgA), self.transform(imgB) 
        else:
            return imgA, imgB
        
    def __len__(self):
        return max(self.len_A, self.len_B)


# define intial weight function
def init_weights(model):
    classname = model.__class__.__name__
    if hasattr(model, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1) :
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)


# define hyperparamter
LR = 0.0002
BATCH_SIZE = 1
N_EPOCHES = 200
LAMBDA = 10
D_RATIO = 2
N_BLOCKS = 9
LR_LAMBDA = lambda epoch: min(1, 1 - (epoch-100) / 100)
IMG_SIZE = 286
INPUT_SIZE = 256


# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Networks
G_A2B = Generator(input_dim=3, n_blocks=N_BLOCKS).to(device)
G_B2A = Generator(input_dim=3, n_blocks=N_BLOCKS).to(device)
D_A = Discriminator(input_dim=3).to(device)
D_B = Discriminator(input_dim=3).to(device)
G_A2B.apply(init_weights)
G_B2A.apply(init_weights)
D_A.apply(init_weights)
D_B.apply(init_weights)

# ImagePool
fake_A_pool = ImagePool(size=50)
fake_B_pool = ImagePool(size=50)

# loss
Loss_GAN = nn.MSELoss()
Loss_cyc = nn.L1Loss()

# optimizer , betas=(0.5, 0.999)
optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=LR, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LR_LAMBDA)
scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LR_LAMBDA)
scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LR_LAMBDA)

# transform
train_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), 
                                      transforms.RandomCrop((INPUT_SIZE, INPUT_SIZE)), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
test_transform = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC), 
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

# dataset
trainset = ImageFolder(root_dir=root, mode='train', transform=train_transform)
testset = ImageFolder(root_dir=root, mode='test', transform=test_transform)

# dataloader
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# training process
train_D_As = []
train_D_Bs= []
train_G_A2Bs = []
train_G_B2As = []
test_D_As = []
test_D_Bs = []
test_G_A2Bs = []
test_G_B2As = []

for epo in tqdm(range(N_EPOCHES)):
    # train_loader
    train_D_A = []
    train_D_B = []
    train_G_A2B = []
    train_G_B2A = []
    G_A2B.train()
    G_B2A.train() 
    D_A.train()
    D_B.train() 
    for real_A, real_B in train_loader:
        
        real_A, real_B = real_A.to(device), real_B.to(device)
        n_img = len(real_A)
        
        ##### Generator
        # A --> B
    
        fake_B = G_A2B(real_A)
        fake_B2A = G_B2A(fake_B)
        eval_fake_B = D_B(fake_B)
        
        G_A2B_loss =  Loss_GAN(eval_fake_B, torch.ones_like(eval_fake_B).to(device))
        cycleA_loss = Loss_cyc(fake_B2A, real_A)

        # B --> A
        fake_A = G_B2A(real_B)
        fake_A2B = G_A2B(fake_A)
        eval_fake_A = D_A(fake_A)

        G_B2A_loss =  Loss_GAN(eval_fake_A, torch.ones_like(eval_fake_A).to(device))
        cycleB_loss = Loss_cyc(fake_A2B, real_B)
       
        full_loss = G_A2B_loss + G_B2A_loss + LAMBDA * (cycleA_loss + cycleB_loss)
        optimizer_G.zero_grad()
        full_loss.backward()
        optimizer_G.step()
        #torch.cuda.empty_cache()
        
        ##### Discriminator A
        fake_A = fake_A_pool.draw(fake_A).to(device)
        eval_real_A = D_A(real_A)
        eval_fake_A = D_A(fake_A)       
        
        eval_real_A_loss = Loss_GAN(eval_real_A, torch.ones_like(eval_real_A).to(device))
        eval_fake_A_loss = Loss_GAN(eval_fake_A, torch.zeros_like(eval_fake_A).to(device))
        evalA_loss = (eval_real_A_loss + eval_fake_A_loss) / D_RATIO
    
        optimizer_D_A.zero_grad()
        evalA_loss.backward()
        optimizer_D_A.step()
        #torch.cuda.empty_cache()
        
        ##### Discriminator B
        fake_B = fake_B_pool.draw(fake_B).to(device)
        eval_real_B = D_B(real_B)
        eval_fake_B = D_B(fake_B)
             
        eval_real_B_loss = Loss_GAN(eval_real_B, torch.ones_like(eval_real_B).to(device))
        eval_fake_B_loss = Loss_GAN(eval_fake_B, torch.zeros_like(eval_fake_B).to(device))
        evalB_loss = (eval_real_B_loss + eval_fake_B_loss) / D_RATIO

        optimizer_D_B.zero_grad()
        evalB_loss.backward()
        optimizer_D_B.step()
        #torch.cuda.empty_cache()
        
        train_D_A.append(evalA_loss.item())
        train_D_B.append(evalB_loss.item())
        train_G_A2B.append(G_A2B_loss.item())
        train_G_B2A.append(G_B2A_loss.item())

        torch.cuda.empty_cache()
    scheduler_G.step()
    scheduler_D_A.step()
    scheduler_D_B.step()
    print('Epoch {0:2d}/{1}:'.format(epo+1, N_EPOCHES))
    mean_train_DA = np.mean(train_D_A)
    mean_train_DB = np.mean(train_D_B)
    mean_train_G_A2B = np.mean(train_G_A2B)
    mean_train_G_B2A = np.mean(train_G_B2A)
    
    train_D_As.append(mean_train_DA)
    train_D_Bs.append(mean_train_DB)
    train_G_A2Bs.append(mean_train_G_A2B)
    train_G_B2As.append(mean_train_G_B2A)
    print('Training : D_A_loss: {0:.4f}, D_B_loss: {1:.4f}, G_A2B_loss: {2:.4f}, G_B2A_loss: {3:.4f}'.
          format(mean_train_DA, mean_train_DB, mean_train_G_A2B, mean_train_G_B2A))
                           
    # test_loader
    test_D_A = []
    test_D_B = []
    test_G_A2B = []
    test_G_B2A = []
    G_A2B.eval()
    G_B2A.eval() 
    D_A.eval()
    D_B.eval() 
    with torch.no_grad():
        for real_A, real_B in test_loader:
            real_A, real_B = real_A.to(device), real_B.to(device)
            n_img = len(real_A)
        
            ##### Generator
            # A --> B

            fake_B = G_A2B(real_A)
            fake_B2A = G_B2A(fake_B)
            eval_fake_B = D_B(fake_B)

            G_A2B_loss =  Loss_GAN(eval_fake_B, torch.ones_like(eval_fake_B).to(device))
            cycleA_loss = Loss_cyc(fake_B2A, real_A)

            # B --> A
            fake_A = G_B2A(real_B)
            fake_A2B = G_A2B(fake_A)
            eval_fake_A = D_A(fake_A)

            G_B2A_loss =  Loss_GAN(eval_fake_A, torch.ones_like(eval_fake_A).to(device))
            cycleB_loss = Loss_cyc(fake_A2B, real_B)

            full_loss = G_A2B_loss + G_B2A_loss + LAMBDA * (cycleA_loss + cycleB_loss)

            #torch.cuda.empty_cache()

            ##### Discriminator A
            fake_A = fake_A_pool.draw(fake_A).to(device)
            eval_real_A = D_A(real_A)
            eval_fake_A = D_A(fake_A)       

            eval_real_A_loss = Loss_GAN(eval_real_A, torch.ones_like(eval_real_A).to(device))
            eval_fake_A_loss = Loss_GAN(eval_fake_A, torch.zeros_like(eval_fake_A).to(device))
            evalA_loss = (eval_real_A_loss + eval_fake_A_loss) / D_RATIO
            #torch.cuda.empty_cache()

            ##### Discriminator B
            fake_B = fake_B_pool.draw(fake_B).to(device)
            eval_real_B = D_B(real_B)
            eval_fake_B = D_B(fake_B)

            eval_real_B_loss = Loss_GAN(eval_real_B, torch.ones_like(eval_real_B).to(device))
            eval_fake_B_loss = Loss_GAN(eval_fake_B, torch.zeros_like(eval_fake_B).to(device))
            evalB_loss = (eval_real_B_loss + eval_fake_B_loss) / D_RATIO

            test_D_A.append(evalA_loss.item())
            test_D_B.append(evalB_loss.item())
            test_G_A2B.append(G_A2B_loss.item())
            test_G_B2A.append(G_B2A_loss.item())
        mean_test_DA = np.mean(test_D_A)
        mean_test_DB = np.mean(test_D_B)
        mean_test_G_A2B = np.mean(test_G_A2B)
        mean_test_G_B2A = np.mean(test_G_B2A)
        test_D_As.append(mean_test_DA)
        test_D_Bs.append(mean_test_DB)
        test_G_A2Bs.append(mean_test_G_A2B)
        test_G_B2As.append(mean_test_G_B2A)
        print('Testing  : D_A_loss: {0:.4f}, D_B_loss: {1:.4f}, G_A2B_loss: {2:.4f}, G_B2A_loss: {3:.4f}'.
              format(mean_test_DA, mean_test_DB, mean_test_G_A2B, mean_test_G_B2A))


# ### Evaluation
import matplotlib.pyplot as plt

# training part
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.plot(train_D_As)
plt.title('train_D_As, LAMBDA = {0}, D_RATIO = {1}'.format(LAMBDA, D_RATIO))
plt.subplot(2,2,2)
plt.plot(train_D_Bs)
plt.title('train_D_Bs, LAMBDA = {0}, D_RATIO = {1}'.format(LAMBDA, D_RATIO))
plt.subplot(2,2,3)
plt.plot(train_G_A2Bs)
plt.title('train_G_A2Bs, LAMBDA = {0}, D_RATIO = {1}'.format(LAMBDA, D_RATIO))
plt.subplot(2,2,4)
plt.plot(train_G_B2As)
plt.title('train_G_B2As, LAMBDA = {0}, D_RATIO = {1}'.format(LAMBDA, D_RATIO))
plt.show()

# testing part
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.plot(test_D_As)
plt.title('test_D_As, LAMBDA = {0}, D_RATIO = {1}'.format(LAMBDA, D_RATIO))
plt.subplot(2,2,2)
plt.plot(test_D_Bs)
plt.title('test_D_Bs, LAMBDA = {0}, D_RATIO = {1}'.format(LAMBDA, D_RATIO))
plt.subplot(2,2,3)
plt.plot(test_G_A2Bs)
plt.title('test_G_A2Bs, LAMBDA = {0}, D_RATIO = {1}'.format(LAMBDA, D_RATIO))
plt.subplot(2,2,4)
plt.plot(test_G_B2As)
plt.title('test_G_B2As, LAMBDA = {0}, D_RATIO = {1}'.format(LAMBDA, D_RATIO))
plt.show()


import pickle

losses = {'train_D_As': train_D_As,
          'train_D_Bs': train_D_Bs,
          'train_G_A2Bs': train_G_A2Bs,
          'train_G_B2As': train_G_B2As,
          'test_D_As': test_D_As,
          'test_D_Bs': test_D_Bs,
          'test_G_A2Bs': test_G_A2Bs,
          'test_G_B2As': test_G_B2As}


pickle.dump(losses, open('losses/losses_L{0}_D{1}_version3.pkl'.format(LAMBDA, D_RATIO), 'wb'))
torch.save(G_A2B, 'models/G_A2B_L{0}_D{1}_version3.pkl'.format(LAMBDA, D_RATIO))
torch.save(G_B2A, 'models/G_B2A_L{0}_D{1}_version3.pkl'.format(LAMBDA, D_RATIO))


os.mkdir('Test_Result/L{0}_D{1}'.format(LAMBDA, D_RATIO))
os.mkdir('Test_Result/L{0}_D{1}/Domain_A'.format(LAMBDA, D_RATIO))
os.mkdir('Test_Result/L{0}_D{1}/Domain_B'.format(LAMBDA, D_RATIO))


# convert tensor to image
def tensor2im(input_image, imtype=np.uint8):

    image_tensor = input_image.data

    image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    #image_numpy = (((image_numpy - image_numpy.min()) * 255) / (image_numpy.max() - image_numpy.min())).transpose(1, 2, 0).astype(np.uint8)
    return image_numpy.astype(imtype)


# save testing image
cnt = 0
for real_A, real_B in tqdm(test_loader):
    
    real_A, real_B = real_A.to(device), real_B.to(device)
    
    fake_B = G_A2B(real_A)
    fake_B2A = G_B2A(fake_B)

    fake_A = G_B2A(real_B)
    fake_A2B = G_A2B(fake_A)
    
    plt.figure(figsize=(12,10))
    plt.subplot(1,3,1)
    plt.title('real_A')
    plt.imshow(tensor2im(real_A))
    plt.subplot(1,3,2)
    plt.title('fake_B')
    plt.imshow(tensor2im(fake_B))
    plt.subplot(1,3,3)
    plt.title('fake_B2A')
    plt.imshow(tensor2im(fake_B2A))
    plt.savefig('Test_Result/L{0}_D{1}/Domain_{2}/{3}.jpg'.format(LAMBDA, D_RATIO, 'A', cnt))   # save the figure to file
    plt.close()
    
    plt.figure(figsize=(12,10))
    plt.subplot(1,3,1)
    plt.title('real_B')
    plt.imshow(tensor2im(real_B))
    plt.subplot(1,3,2)
    plt.title('fake_A')
    plt.imshow(tensor2im(fake_A))
    plt.subplot(1,3,3)
    plt.title('fake_A2B')
    plt.imshow(tensor2im(fake_A2B))
    plt.savefig('Test_Result/L{0}_D{1}/Domain_{2}/{3}.jpg'.format(LAMBDA, D_RATIO, 'B', cnt))   # save the figure to file
    plt.close()
    
    cnt += 1

# define function to compare result
def show_3_plot(real, fake, reconstruct):
    plt.figure(figsize=(12,10))
    plt.subplot(1,3,1)
    plt.imshow(tensor2im(real, imtype=np.uint8))
    plt.subplot(1,3,2)
    plt.imshow(tensor2im(fake, imtype=np.uint8))
    plt.subplot(1,3,3)
    plt.imshow(tensor2im(reconstruct, imtype=np.uint8))
    

show_3_plot(real_A, fake_B, fake_B2A)
show_3_plot(real_B, fake_A, fake_A2B)


