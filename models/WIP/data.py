import glob
import os
import pandas as pd 
from torch import nn
from sklearn.model_selection import train_test_split
from pandas.api.types import CategoricalDtype

import numpy as np 

import torch
import torchtext
from torchtext.data import get_tokenizer

from collections import Counter
MAX_LEN = 200
TRAIN_BATCH_SIZE = 124
VALID_BATCH_SIZE = 124
EPOCHS = 25
LEARNING_RATE = 1e-01

class Simple_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_tokens):
        self.df = df
        self.vocab = vocab
        self.max_length = max_tokens
        self.tokenizer = get_tokenizer("basic_english")
        
        # label 1 is negative sentiment and label 2 is positive sentiment
        self.label_map = {1:0, 2:1}
        
    def __len__(self):
        return len(self.df)

    def decode(self, numericalized_tokens):
        return ' '.join([self.vocab.get_itos()[num] for num in numericalized_tokens])

    def __getitem__(self, index):
        text, label = self.df.iloc[index]
        label = self.label_map[label]
        label = torch.tensor(label)

        tokens = tokenizer(text)
        numericalized_tokens = [my_vocab[token] for token in tokens]

        if len(numericalized_tokens) < self.max_length:
            numericalized_tokens += [1] * (self.max_length-len(numericalized_tokens))
        else:
            numericalized_tokens = numericalized_tokens[:self.max_length]

        numericalized_tokens = torch.tensor(numericalized_tokens)
        
        return numericalized_tokens, label

class Preprocessing:
    def __init__(self, data_path, test_path, seed=0):
        self.data_path = data_path
        self.test_path = test_path
        self.max_len = 10
        self.max_words = 100
        self.test_size = .3
        # self.val_size = .2
        self.cat_type = CategoricalDtype(
            categories=["not_informative", "informative"], ordered=True)
        self.seed = seed

    def load_data(self):
        df = self.preprocess(self.data_path)
        
        self.df_test = self.preprocess(self.test_path, True)

        self.df_train, self.df_val = train_test_split(
            df, test_size= self.test_size, random_state= self.seed)

    def preprocess(self, data_path, is_test = False):
        if not is_test:
            df = pd.read_csv(data_path, sep = '\t')
            df = df[['tweet_text', 'text_info']]

            df.rename(columns= {'tweet_text': 'text', 'text_info': 'label'}, inplace=True)

            match = 'informative'

        else:
            df = pd.read_csv(data_path)
            df = df[[' Tweet Text', ' Informativeness']]

            df.rename(columns= {' Tweet Text': 'text', ' Informativeness': 'label'}, inplace=True)


            match = 'Related and informative'

        mask = df['label'] == match
        df.label[mask] = 2
        df.label[~mask] = 1

        return df


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.CNN.data import preprocess_image

class DatasetMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y 

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ImageTextMapper(Dataset):
    def __init__(self, text, image, y):
        self.text = text
        self.image = image
        self.y = y 

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        image = preprocess_image(self.image[idx])
        return self.text[idx], image, self.y[idx]


# training_set = DatasetMapper(d.df_train.text, d.df_train.target)
# validation_set = DataMapper(d.df_val.text, d.df_val.target)
# test_set = DatasetMapper(d.df_test.text, d.df_test.target)

# loader_training = DataLoader(training_set, batch_size=batch_size)
# loader_validation = DataLoader(training_set, batch_size=batch_size)
# loader_test = DataLoader(test_set)

# data_path = './CrisisMMD_v2.0/annotations/'
# token_counter = Counter()
# tokenizer = get_tokenizer("basic_english")

# for crisis in glob.glob(os.path.join('./CrisisMMD_v2.0/annotations/', '*')):
#     df = Preprocessing(crisis, '/home/infres/ralghoss-21/Crisis/CrisisLexT26/2012_Colorado_wildfires/2012_Colorado_wildfires-tweets_labeled.csv')
#     df.load_data()


#     for sample_text in df.df_train['text']:
#         tokens = tokenizer(sample_text)
#         token_counter.update(tokens)

#     break

# sorted_counter = dict(token_counter.most_common())

# # Create vocab containing tokens with a minimum frequency of 20
# my_vocab = torchtext.vocab.vocab(sorted_counter, min_freq=3)

# my_vocab.insert_token('', 0)
# my_vocab.set_default_index(0)

# glove = torchtext.vocab.GloVe(name = '840B', dim = 200)
# my_vocab.vectors = glove.get_vecs_by_tokens(my_vocab.get_itos())

# tot_transferred = 0
# for v in my_vocab.vectors:
#     if not v.equal(torch.zeros(200)):
#         tot_transferred += 1
        
# for i in range(my_vocab.vectors.shape[0]):
#     if my_vocab.vectors[i].equal(torch.zeros(200)):
#         my_vocab.vectors[i] = torch.randn(200)


# train_dataset = Simple_Dataset(df.df_train, vocab=my_vocab, max_tokens=128)
# valid_dataset = Simple_Dataset(df.df_val, vocab=my_vocab, max_tokens=128)
# test_dataset = Simple_Dataset(df.df_test, vocab=my_vocab, max_tokens=128)

train_total = []
test_total = []

for crisis in glob.glob(os.path.join('./CrisisMMD_v2.0/annotations/', '*')):
    df = pd.read_csv(crisis, sep = '\t')
    print(crisis)
    df = df[['tweet_text', 'text_info']]

    df.rename(columns= {'tweet_text': 'text', 'text_info': 'label'}, inplace=True)
    match = 'informative'
    mask = df['label'] == match
    df.label[mask] = 1
    df.label[~mask] = 0

    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    training_set = DatasetMapper(train_dataset.text, train_dataset.label)
    testing_set = DatasetMapper(test_dataset.text, test_dataset.label)


    train_total.append(training_set)
    test_total.append(testing_set)
    break

# print(len(train_dataset[train_dataset.label == 1])/len(train_dataset))
# print(len(test_dataset[test_dataset.label == 1])/len(test_dataset))

training_set = torch.utils.data.ConcatDataset(train_total)
testing_set = torch.utils.data.ConcatDataset(test_total)


# print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(len(training_set)))
print("TEST Dataset: {}".format(len(testing_set)))




training_loader = DataLoader(training_set, batch_size = TRAIN_BATCH_SIZE)
testing_loader = DataLoader(testing_set, batch_size = VALID_BATCH_SIZE)