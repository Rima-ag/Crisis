import ART1 as art1
from art1.data import feature_as_cluster
from art1.model import ART1
import numpy as np

paths = ['./california_wildfires_final_data.tsv', './hurricane_harvey_final_data.tsv',\
          './srilanka_floods_final_data.tsv', './hurricane_maria_final_data.tsv',\
         './iraq_iran_earthquake_final_data.tsv', './mexico_earthquake_final_data.tsv']

leave_one_out = './hurricane_irma_final_data.tsv'

df = feature_as_cluster(paths, leave_one_out)

X, y = df[['text']].to_numpy(), df[['label']].to_numpy()

net = ART1(df.shape[-1])
labels = net.predict(X)

diff = np.square(np.subtract(labels, y.ravel()))
_, accuracy = np.unique(diff, return_counts = True)

print(f"Accuracy: {accuracy[0] * 100 / accuracy.sum()}")