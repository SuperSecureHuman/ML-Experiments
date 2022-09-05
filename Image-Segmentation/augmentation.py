import os
import time
import albumentations as A
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

# Create list of images and mask

def create_list(images_path, masks_path):
    images = []
    masks = []
    for filename in os.listdir(images_path):
        images.append(os.path.join(images_path, filename))
    for filename in os.listdir(masks_path):
        masks.append(os.path.join(masks_path, filename))
    print(len(images), len(masks))
    return images, masks

def augmentation(image, mask):
    # Create augmentation pipeline

    image = Image.open(image).convert('RGB')
    image = np.array(image)  

    mask = Image.open(mask).convert('L')
    mask = np.array(mask)


    #aug = A.RandomRotate90(p=1.0)
    #augmented = aug(image=image, mask=mask)
    #i1 = augmented['image']
    #m1 = augmented['mask']

    aug = A.HorizontalFlip(p=1.0)
    augmented = aug(image=image, mask=mask)
    i2 = augmented['image']
    m2 = augmented['mask']

    aug = A.VerticalFlip(p=1.0)
    augmented = aug(image=image, mask=mask)
    i3 = augmented['image']
    m3 = augmented['mask']

    aug = A.GridDistortion(p=1.0)
    augmented = aug(image=image, mask=mask)
    i4 = augmented['image']
    m4 = augmented['mask']

    augmented_image_list = [image, i2, i3, i4]
    augmented_mask_list = [mask, m2, m3, m4]

    return augmented_image_list, augmented_mask_list

def save_list_images(input_list, output_path, prefix):
    
    for i in range(len(input_list)):
        fileName = prefix + '__' + str(i) + '.png'
        file_pth = os.path.join(output_path, fileName)
        cv2.imwrite(file_pth, input_list[i])


if not os.path.exists('./augmented_images'):
    os.makedirs('./augmented_images')
if not os.path.exists('./augmented_masks'):
    os.makedirs('./augmented_masks')


image_list, mask_list = create_list('./images', './masks')

image_len = len(image_list)

for i in tqdm(range(image_len), desc='Augmenting images', unit='images', leave=False, position=0, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
    prefix = image_list[i].split('/')[-1].split('.')[0]
    augmented_image_list, augmented_mask_list = augmentation(image_list[i], mask_list[i])
    save_list_images(augmented_image_list, './augmented_images', prefix)
    save_list_images(augmented_mask_list, './augmented_masks', prefix)

