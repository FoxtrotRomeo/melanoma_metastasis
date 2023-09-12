import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl

from resources.model import SweDeClinBert, BiLSTM
from resources.data import generate_data

NUM_EPOCHS = 20
BATCH_SIZE = 2
LABEL      = 'metastasis'
RNN        = BiLSTM

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# load training data:
data_train = pd.read_csv('data/text_train_set.csv', sep='\t')
data_train.dropna(inplace=True)
data_train

# calculate class weights:
y = np.array([data_train[LABEL].values[data_train.pid==pid][0] for pid in data_train.pid.unique()])

neg, pos = np.bincount(np.ravel(y.astype(int)))
total = neg + pos

print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

CLASS_WEIGHTS = 1 / np.array((neg,pos)) * (total / 2.0)
print('Weight for class 0:   {:.2f}'.format(CLASS_WEIGHTS[0]))
print('Weight for class 1:   {:.2f}'.format(CLASS_WEIGHTS[1]))

INITIAL_BIAS = np.log([pos/neg])[0]
print('\nInitial bias:         {:.2f}'.format(INITIAL_BIAS))

ds_train = generate_data(data_train, label=LABEL, batch_size=BATCH_SIZE, n_prefetch=1, class_weights=CLASS_WEIGHTS)

# load validation data:
data_valid = pd.read_csv('data/text_val_set.csv', sep='\t')
data_valid.dropna(inplace=True)
data_valid

ds_valid = generate_data(data_valid, label=LABEL, batch_size=BATCH_SIZE, n_prefetch=1)

# create model:
MODEL = SweDeClinBert(1, activation=tf.nn.sigmoid, rnn=RNN)
MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy()
)

history = MODEL.fit(ds_train,
    validation_data=ds_valid,
    epochs=NUM_EPOCHS,
    callbacks=[
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr if epoch < 0.1 * NUM_EPOCHS else lr * 0.1
        ),
        tf.keras.callbacks.EarlyStopping(
            restore_best_weights=True,
            monitor='val_loss',
            mode='min',
            patience=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'model/TensorFlow/checkpoint/SweDeClin-BERT-{RNN.__name__}',
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
    ]
)

# save history:
with open(f'results/SweDeClin-BERT-{RNN.__name__}/train_history.pkl', 'wb') as f:
    pkl.dump(history.history, f)

# save model:
MODEL.save_weights(f'model/TensorFlow/final/SweDeClin-BERT-{RNN.__name__}/weights')

MODEL.summary()
