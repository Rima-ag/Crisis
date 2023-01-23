import torchtext
import ntlk
import pandas as pd 
import re
import numpy as np
import torch

from nltk.corpus import stopwords
nltk.download('stopwords')

from collections import Counter, OrderedDict
from torchtext.data import get_tokenizer
import sknetwork
from sknetwork.clustering import Louvain
from scipy import sparse


tokenizer = get_tokenizer("basic_english")

def _preprocess(data):
  """
  Preprocesses text

  Parameters
  ----------
  data : DataFrame
      should contain a text column that will be preprocessed
  """
  data['text'] = data['text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
  data['text'] = data['text'].map(lambda x: x.lower())
  data['text'] = data['text'].map(lambda x: " ".join([t for t in x.split(" ") if t not in stopwords.words('english')]))
  data['text'] = data['text'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
  data['text'] = data['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
  data = data[data.duplicated('text')==False]
  return data

def _read_data(path):
  """
  Reads data from file and preprocesses the text data

  Parameters:
  ----------
  path : str,
      path to the file containing the dataset
  """

  df = pd.read_csv(path, sep='\t')
  df = df[['tweet_text', 'text_info']]

  df = df.drop(columns=['text_info'])
  df = df.rename(columns={'tweet_text': 'text'})
  return _preprocess(df)

def _gen_vocab(paths):
  """
  Generates vocabulary from a dataset

  Parameters:
  ----------
  paths : list(str),
       paths to the files of the dataset
  """
  glove = torchtext.vocab.GloVe(name = '6B', dim = 100)
  token_counter = Counter()
  for path in paths:
    data = _read_data(path)
    for text in data['text']:
      tokens = tokenizer(text)
      token_counter.update(tokens)

  sorted_counter = dict(token_counter.most_common())

  vocab = torchtext.vocab.vocab(sorted_counter, min_freq=30)

  vocab.insert_token('', 0)
  vocab.set_default_index(0)
  
  vocab.vectors = glove.get_vecs_by_tokens(vocab.get_itos())
  return vocab

def _gen_clusters(vocab):
  """
  Clusters similar words in vocabulary

  Parameters:
  ----------
  vocab : torchtext.vocab.Vocab,
       Vocabulary of words that will be clustered

  Returns:
  -------
  list
       a list of labels for each word in the vocabulary
  """
  def build_adjacency(vocab, op):
    adjacency = np.zeros((len(vocab.get_itos()), len(vocab.get_itos())))

    for i in range(len(vocab.vectors)):
      for j in range(i, len(vocab.vectors)):
        adjacency[i][j] = op(vocab.vectors[i], vocab.vectors[j])
        adjacency[i][j] = 0 if adjacency[i][j] < 0 else adjacency[i][j]
        adjacency[j][i] = adjacency[i][j]

    return adjacency


  cos = torch.nn.CosineSimilarity(dim=0)
  adjacency = sparse.csr_matrix(build_adjacency(vocab, cos))
  labels = Louvain().fit_transform(adjacency)

  return labels

def feature_as_cluster(paths, leave_one_out):
  """
  Builds new feature space for a dataset built using a vocabulary.
  
  Parameters:
  ----------
  paths : list(str),
       datasets that will be used to build feature space

  leave_one_out : str
        dataset that will be mapped to feature space
  
  Returns:
  -------
  DataFrame
        dataframe with new features representing properties in text
  """
  vocab = _gen_vocab(paths)
  labels = _gen_clusters(vocab)

  df = pd.read_csv(leave_one_out, sep='\t')
  df = df[['tweet_text', 'text_info']]

  df = df.rename(columns={'tweet_text': 'text', 'tweet_info': 'label'})
  df = _preprocess(df)
  mask = df[['text_info']]  == 'informative'
  df[mask] = 1
  df[~mask] = 0

  labels_unique, counts = np.unique(labels, return_counts=True)

  for i in labels_unique:
    if counts[i] > 1:
      df[str(i)] = df['text'].map(lambda x : 1 if len([t for t in x.split(" ") if t in np.array(vocab.get_itos())[labels == i]])!= 0 else 0)

  df = df.drop(columns=['text'])
  return df