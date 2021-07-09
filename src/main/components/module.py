import os

from typing import Union

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_transform as tft

LABEL_KEY = 'flow flood'

# Feature: float
# Kiem tra xem min va max co xa nhau ko, neu co thi ghi vao day
DENSE_FLOAT_FEATURE_KEYS = {
    'flow downstream': None,
    'flow lake': None,
    'height downstream': None,
    'height lake': None
}

def transformed_name(key):
  return key + '_xf'

def fill_in_missing(x):
  if isinstance(x, tf.sparse.SparseTensor):
    default_value = "" if x.dtype == tf.string else 0
    x = tf.sparse.to_dense(
        tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1])
    )
  return tf.squeeze(x, axis=1)

def convert_num_to_one_hot(label_tensor, num_labels=2):
  one_hot_tensor = tf.one_hot(label_tensor, num_labels)
  return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
  outputs = {}

  for key in DENSE_FLOAT_FEATURE_KEYS.keys():
    # Preserve this feature as a dense float, setting nan's to the man
    outputs[transformed_name(key)] = tft.scale_to_z_score(
        fill_in_missing(inputs[key])
    )
  
  outputs[transformed_name(LABEL_KEY)] = fill_in_missing(inputs[LABEL_KEY])
  return outputs