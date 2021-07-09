from typing import List, Text
import os
import absl
import datetime
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
import components.module as tc

def _get_serve_tf_examples_fn(model, tf_transform_output):
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(tc.LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)
  return serve_tf_examples_fn

def transformed_name(key):
  return key + '_xf'

def get_model():
  # One-hot categorical features
  real_valued_columns = []
  for key in tc.DENSE_FLOAT_FEATURE_KEYS.keys():
    real_valued_columns.append(
        tf.keras.Input(shape=(1,),
                       name=transformed_name(key),
                       dtype=tf.float32)
    )
  inputs = real_valued_columns
  first_dnn_layer_size = 100
  num_dnn_layers = 4
  dnn_decay_factor = 0.7
  hidden_units = [
                  max(2, int(first_dnn_layer_size*dnn_decay_factor**i))
                  for i in range(num_dnn_layers)
  ]
  deep_ff = tf.keras.layers.concatenate(real_valued_columns)
  for numnodes in hidden_units:
    deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep_ff)
  output = tf.keras.layers.Dense(1, activation='sigmoid')(deep)
  keras_model = tf.keras.models.Model(inputs, output)
  keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=[
                               tf.keras.metrics.BinaryAccuracy()
                      ])
  return keras_model

def _gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=40):
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy()
  )

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=transformed_name(tc.LABEL_KEY)
  )
  return dataset

def run_fn(fn_args):
  tf_transform_output = tft.TFTransformOutput(
      fn_args.transform_output
  )
  train_dataset = input_fn(fn_args.train_files, tf_transform_output)
  eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)
  model = get_model()
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir =log_dir, update_freq='batch'
  )
  model.fit(
      train_dataset,
      steps_per_epoch = fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback]
  )
  signatures = {
      'serving_default':
      _get_serve_tf_examples_fn(model,tf_transform_output).get_concrete_function(
          tf.TensorSpec(
              shape=[None],
              dtype=tf.string,
              name='examples'
          )
      )
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)