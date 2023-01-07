import torchtext
import ntlk
import pandas as pd 
import re
import numpy as np
import torch

from nltk.corpus import stopwords
nltk.download('stopwords')

from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torchtext.data import get_tokenizer
import sknetwork
from sknetwork.clustering import Louvain
from scipy import sparse


tokenizer = get_tokenizer("basic_english")

def preprocess(data):
  data['text'] = data['text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
  data['text'] = data['text'].map(lambda x: x.lower())
  data['text'] = data['text'].map(lambda x: " ".join([t for t in x.split(" ") if t not in stopwords.words('english')]))
  data['text'] = data['text'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
  data['text'] = data['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
  data = data[data.duplicated('text')==False]
  return data

def read_data(path):

  df = pd.read_csv(path, sep='\t')
  df = df[['tweet_text', 'text_info']]

  df = df.drop(columns=['text_info'])
  df = df.rename(columns={'tweet_text': 'text'})
  return preprocess(df)

def gen_vocab(paths):
    glove = torchtext.vocab.GloVe(name = '6B', dim = 100)
    token_counter = Counter()
    for path in paths:
        data = read_data(path)
        for text in data['text']:
            tokens = tokenizer(text)
            token_counter.update(tokens)

    sorted_counter = dict(token_counter.most_common())

    vocab = torchtext.vocab.vocab(sorted_counter, min_freq=30)

    vocab.insert_token('', 0)
    vocab.set_default_index(0)
    
    vocab.vectors = glove.get_vecs_by_tokens(my_vocab.get_itos())
    return vocab

def gen_clusters(vocab):
    adjacency = np.zeros((len(my_vocab.get_itos()), len(my_vocab.get_itos())))
    cos = torch.nn.CosineSimilarity(dim=0)

    for i in range(len(my_vocab.vectors)):
        for j in range(i, len(my_vocab.vectors)):
            adjacency[i][j] = cos(my_vocab.vectors[i], my_vocab.vectors[j])
            adjacency[i][j] = 0 if adjacency[i][j] < 0 else adjacency[i][j]
            adjacency[j][i] = adjacency[i][j]

    adjacency = sparse.csr_matrix(adjacency)
    louvain = Louvain()
    labels = louvain.fit_transform(adjacency)
    labels_unique, counts = np.unique(labels, return_counts=True)
    return labels

def feature_as_cluster(paths, leave_one_out):
    vocab = gen_vocab(paths)
    labels = gen_clusters(vocab)

    df = pd.read_csv(leave_one_out, sep='\t')
    df = df[['tweet_text', 'text_info']]

    df = df.rename(columns={'tweet_text': 'text', 'tweet_info': 'label'})
    df = preprocess(df)
    mask = df[['text_info']]  == 'informative'
    df[mask] = 1
    df[~mask] = 0

    labels_unique, counts = np.unique(labels, return_counts=True)

    for i in labels_unique:
        if counts[i] > 1:
            df[str(i)] = df['text'].map(lambda x : 1 if len([t for t in x.split(" ") if t in np.array(vocab.get_itos())[labels == i]])!= 0 else 0)

    df = df.drop(columns=['text'])
    return df