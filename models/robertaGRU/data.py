import glob
import os
import pandas as pd 
import numpy as np 
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-01

class TextMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y 

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def preprocess_data(df, path):
    df = df[['tweet_text', 'text_info']]

    df.rename(columns= {'tweet_text': 'text', 'text_info': 'label'}, inplace=True)
    match = 'informative'
    mask = df['label'] == match
    df.label[mask] = 1
    df.label[~mask] = 0

    return df

# Will read one dataset only, not enough resources to train and test on all the datasets
def get_dataset(path= './CrisisMMD_v2.0/'):

    train_total = []
    test_total = []

    for crisis in glob.glob(os.path.join('./CrisisMMD_v2.0/annotations/', '*')):
        df = pd.read_csv(crisis, sep = '\t')
        print(crisis)
        df = preprocess_data(df, path)

        train_size = 0.7
        train_dataset=df.sample(frac=train_size,random_state=200)
        test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)
        training_set = TextMapper(train_dataset.text, train_dataset.label)
        testing_set = TextMapper(test_dataset.text, test_dataset.label)


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




