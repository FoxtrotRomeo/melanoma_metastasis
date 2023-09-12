import pandas as pd
import tensorflow as tf
from typing import Union, Iterable

def generate_data(data:pd.DataFrame, label:str, batch_size:int, n_prefetch:int, class_weights:Union[Iterable[float],None]=None, include_labels:bool=True):
    pids = data.pid.unique()
    x = tf.ragged.constant(
        [data.evalue.values[data.pid==pid] for pid in pids],
        name='texts',
        dtype=tf.string
    )
    x = x[:,::-1].to_tensor('')[:,::-1]

    if not include_labels:
        return tf.data.Dataset.from_tensor_slices(x).batch(batch_size).prefetch(n_prefetch)

    y = tf.constant(
        [data[label].values[data.pid==pid][0] for pid in pids],
        name='label',
        dtype=tf.int32
    )

    if class_weights is None:
        return tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(x),
            tf.data.Dataset.from_tensor_slices(y)
        )).batch(batch_size).prefetch(n_prefetch)

    w = tf.constant(
        [class_weights[data[label].values[data.pid==pid][0]] for pid in pids],
        name='weights',
        dtype=tf.float32
    )

    return tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(x),
            tf.data.Dataset.from_tensor_slices(y),
            tf.data.Dataset.from_tensor_slices(w),
        )).batch(batch_size).prefetch(n_prefetch)
