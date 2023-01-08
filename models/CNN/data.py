import glob
import os
import pandas as pd 
import numpy as np 

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import collections
collections.Iterable = collections.abc.Iterable


TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-03

def preprocess_image(url, transforms):
    image = Image.open(url)
    image = np.array(image)
    if len(image.shape) == 2:
        tmp = np.zeros((3, image.shape[0], image.shape[1]))
        tmp[0] = image
        tmp[1] = image
        tmp[2] = image
        image = tmp
    
    image = image[..., :3]
    
    image = transforms(image)
    return image


class ImageMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]) 

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = preprocess_image(self.x[idx], self.transforms)
        return image, self.y[idx]



def preprocess_data(df, path):
    df = df[['image_path', 'image_info']]

    df.rename(columns= {'image_path': 'image', 'image_info': 'label'}, inplace=True)
    df['image'] = path + df['image'].astype(str)
    match = 'informative'
    mask = df['label'] == match
    df.label[mask] = 1
    df.label[~mask] = 0
    return df

def get_dataset(path = '../../CrisisMMD_v2.0/'):

    train_total = []
    test_total = []

    for crisis in glob.glob(os.path.join(path, 'annotations', '*')):
        df = pd.read_csv(crisis, sep = '\t')
        print(crisis)
        df = preprocess_data(df, path)

        train_size = 0.7
        train_dataset=df.sample(frac=train_size,random_state=200)
        test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)
        training_set = ImageMapper(train_dataset.image, train_dataset.label)
        testing_set = ImageMapper(test_dataset.image, test_dataset.label)


        train_total.append(training_set)
        test_total.append(testing_set)
        break

    training_set = torch.utils.data.ConcatDataset(train_total)
    testing_set = torch.utils.data.ConcatDataset(test_total)


    # print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(len(training_set)))
    print("TEST Dataset: {}".format(len(testing_set)))

    training_loader = DataLoader(training_set, batch_size = TRAIN_BATCH_SIZE)
    testing_loader = DataLoader(testing_set, batch_size = VALID_BATCH_SIZE)


    return training_loader, testing_loader


