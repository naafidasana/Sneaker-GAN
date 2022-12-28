# -*- coding: utf-8 -*-
# Author = Naafi Dasana IBRAHIM
"""Sneaker-GAN.ipynb


Notebook version of script is located at
    https://colab.research.google.com/drive/1Zz6VzZU20mBXdQ9cu-mLRE9pugJ1DSVF
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import os

batch_size = 64
latent_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output.view(-1, 1).squeeze(1)

class Generator(nn.Module):

    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

D = Discriminator().to(device)
D.apply(init_weights)
G = Generator(latent_size).to(device)
G.apply(init_weights)
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.00002, betas=(0.55, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.55, 0.999))
print(D.eval())
print(G.eval())

from google.colab import drive
drive.mount('/gdrive')

root = '/gdrive/My Drive/Photos/Real_Sneakers/'

dataset = ImageFolder(root=root, transform=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def train_discriminator(images):
    # Labels to compute Discriminator loss
    real_labels = torch.ones(batch_size,).to(device)
    fake_labels = torch.zeros(batch_size,).to(device)

    # Compute Discriminator loss on real images
    try:
        outputs = D(images)
        d_real_loss = criterion(outputs.to(device), real_labels)
        real_scores = outputs
    except:
        real_labels = torch.ones(45,).to(device)
        fake_labels = torch.zeros(45,).to(device)
        outputs = D(images)
        d_real_loss = criterion(outputs.to(device), real_labels)
        real_scores = outputs

    # Compute Discriminator loss on fake images
    try:
        z = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_fake_loss = criterion(outputs.to(device), fake_labels)
        fake_scores = outputs
    except:
        z = torch.randn(45, latent_size, 1, 1, device=device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_fake_loss = criterion(outputs.to(device), fake_labels)
        fake_scores = outputs

    # Sum up real and fake losses to get actual discriminator loss
    d_loss = d_real_loss + d_fake_loss

    # Opmtimize parameters
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_scores, fake_scores

def train_generator():
    # Generate fake images
    z = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = G(z)
    labels = torch.ones(batch_size,).to(device)
    g_loss = criterion(D(fake_images).to(device), labels)

    # Optimize parameters
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss, fake_images

# Fixed noise vector
fnv = torch.randn(batch_size, latent_size, 1, 1, device=device)

sample_dir = "/gdrive/My Drive/fake_sneakers"
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def save_fake_images(index):
    fake_images = G(fnv)
    fake_images = fake_images.reshape(fake_images.size(0), 3, 64, 64)
    fake_fname = "fake_sneaker-{0:04d}.png".format(index)
    print("Saving", fake_fname)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=8)


# Save fake_image-0000
save_fake_images(0)

num_epochs = 220
n_steps = len(data_loader)

# Full training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Train discriminator
        d_loss, real_scores, fake_scores = train_discriminator(images.to(device))

        # Train generator
        g_loss, fake_images = train_generator()

    # Print model performance every 5th epoch
    if (epoch+1) % 5 == 0:
        print("Epoch {}/{}\t Step {}/{}".format(epoch+1, num_epochs, i+1, n_steps))
        print("D_Loss: {:.3f}, G_Loss: {:.3f}, D(x): {:.2f}, D(G(z)): {:.2f}"
        .format(d_loss.item(), g_loss.item(), real_scores.mean().item(), fake_scores.mean().item()))

    # Save image generated per epoch
    save_fake_images(epoch+1)

import cv2
from IPython.display import FileLink
vid_fname = "sneaker_gan_training.avi"
sample_dir = "/gdrive/My Drive/fake_sneakers/"
files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if "fake_sneaker" in f]
files.sort()
files

vid_path = os.path.join(sample_dir, vid_fname)
out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MP4V'), 8, (302, 302))
[out.write(cv2.imread(fname)) for fname in files]
out.release()
FileLink("sneaker_gan_training.avi")