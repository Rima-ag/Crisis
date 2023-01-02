import glob
import os
import pandas as pd 

import numpy as np 

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
 
sys.path.insert(0, '/home/infres/ralghoss-21/Crisis/models')

import models.CNN.data as image_data
import models.robertaGRU.data as text_data

TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-03

class DataMapper(Dataset):
    def init_mappers(self):
        self.image_mapper = image_data.ImageMapper(self.images, self.labels)
        self.text_mapper = text_data.TextMapper(self.text, self.labels)

    def __init__(self, text, images, labels):
        self.text = text
        self.images = images
        self.labels = labels
        self.init_mappers()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text, label = self.text_mapper[idx]
        image, _ = self.image_mapper[idx]
        return (image, text), label

def get_dataset(path = '../../CrisisMMD_v2.0/'):

    train_total = []
    test_total = []

    for crisis in glob.glob(os.path.join(path, 'annotations', '*')):
        print(crisis)
        all_data = pd.read_csv(crisis, sep = '\t')
        
        df_image= image_data.preprocess_data(all_data, path)
        df_text = text_data.preprocess_data(all_data, path)

        df = df_image.copy()
        df['text'] = df_text['text']
        df['label'] = ((df['label'] == 1) | (df_text['label'] == 1)).astype(int)

        train_size = 0.2
        train_dataset=df.sample(frac=train_size,random_state=200)
        test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)

        training_set = DataMapper(train_dataset.text, train_dataset.image, train_dataset.label)
        testing_set = DataMapper(test_dataset.text, test_dataset.image, test_dataset.label)


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


