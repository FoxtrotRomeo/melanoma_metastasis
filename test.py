import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl

from resources.model import SweDeClinBert, BiLSTM
from resources.data import generate_data

NUM_EPOCHS = 20
BATCH_SIZE = 2
LABEL      = 'metastasis'
RNN        = BiLSTM #tf.keras.layers.GRU

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# load test data:
data_test = pd.read_csv('data/text_test_set.csv', sep='\t')
data_test.dropna(inplace=True)
data_test

ds_test = generate_data(data_test, label=LABEL, batch_size=BATCH_SIZE, n_prefetch=1, include_labels=False)

# create model:
MODEL = SweDeClinBert(1, activation=tf.nn.sigmoid, rnn=RNN)
MODEL.load_weights(f'model/TensorFlow/final/SweDeClin-BERT-{RNN.__name__}/weights')

# predict test set:
labels      = np.array([data_test[LABEL].values[data_test.pid==pid][0] for pid in data_test.pid.unique()])
predictions = MODEL.predict(ds_test)

# save predictions:
with open(f'results/SweDeClin-BERT-{RNN.__name__}/predictions.pkl', 'wb') as f:
    pkl.dump({'y_pred':predictions, 'y_true':labels}, f)
