#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from unet import unet as UNET
import albumentations as A 
from albumentations.pytorch import ToTensorV2

from utils import (
    check_accuracy,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_predictions
)


# In[2]:


# HyperParams
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 16
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = "../dataset/augmented_train_images"
TRAIN_MASK_DIR = "../dataset/augmented_train_masks"
VAL_IMG_DIR = "../dataset/augmented_test_images"
VAL_MASK_DIR = "../dataset/augmented_test_masks"
image_size = (160,240)


# In[3]:


# Train function

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


# In[4]:


# Transormations

transform = A.Compose(
    [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


# In[5]:


model = UNET(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    transform=transform
)


# In[6]:


if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


# In[7]:


torch.cuda.empty_cache()


# In[16]:


check_accuracy(val_loader, model, device=DEVICE)


# In[8]:


scaler = torch.cuda.amp.GradScaler()


# In[ ]:


for epoch in range(NUM_EPOCHS):
    print("Epoch: {}".format(epoch))
    train_fn(train_loader, model, optimizer, loss_fn, scaler)

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE)

    # print some examples to a folder
    save_predictions(
        val_loader, model, folder="saved_images/", device=DEVICE
    )


# In[ ]:




