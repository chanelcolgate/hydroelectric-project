import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import IPython
import IPython.display
from components.Baseline import Baseline
from components.ResidualWrapper import ResidualWrapper
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


class SingleStepModels():
    MAX_EPOCHS = 1
    
    def __init__(self, column_indices, conv_width: int = 0, single_step_window, 
                 wide_window, conv_window=None, wide_conv_window=None, 
                 label_columns=None, num_features: int = 0):
        self.column_indices = column_indices
        self.conv_width = conv_width
        self.single_step_window = single_step_window
        self.wide_window = wide_window
        self.conv_window = conv_window
        self.wide_conv_window = wide_conv_window
        self.label_columns = label_columns
        self.num_features = num_features
        
    def compile_and_fit(self, model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])
        history = model.fit(window.train, epochs=self.MAX_EPOCHS,
                            validation_data = window.val,
                            callbacks=[early_stopping])
        return history
        
    def startApp(self):
        # Baseline
        baseline = Baseline(label_index=self.column_indices['T (degC)'])
        baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])
        
        val_performance = {}
        performance = {}
        
        val_performance['Baseline'] = baseline.evaluate(self.single_step_window.val)
        performance['Baseline'] = baseline.evaluate(self.single_step_window.test, verbose=0)
        if self.label_columns is not None:
            logging.info("Baseline model on `wide_window`")
            logging.info(f'Input shape: {self.wide_window.example[0].shape}')
            logging.info(f'Output shape: {baseline(self.wide_window.example[0]).shape}')
            self.wide_window.plot(baseline)
        
        # Linear
        if self.label_columns is not None:
            linear = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1)])
            history = self.compile_and_fit(linear, self.single_step_window)
            val_performance['Linear'] = linear.evaluate(self.single_step_window.val)
            performance['Linear'] = linear.evaluate(self.single_step_window.test, verbose=0)
            logging.info("Linear model on `wide_window`")
            logging.info(f'Input shape: {self.wide_window.example[0].shape}')
            logging.info(f'Output shape: {linear(self.wide_window.example[0]).shape}')
            self.wide_window.plot(linear)
        
        # Dense
        dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)])
        
        history = self.compile_and_fit(dense, self.single_step_window)
        IPython.display.clear_output()
        
        val_performance['Dense'] = dense.evaluate(self.single_step_window.val)
        performance['Dense'] = dense.evaluate(self.single_step_window.test, verbose=0)
        if self.label_columns is not None:
            logging.info("Dense model on `wide_window`")
            logging.info(f'Input shape: {self.wide_window.example[0].shape}')
            logging.info(f'Output shape: {dense(self.wide_window.example[0]).shape}')
            self.wide_window.plot(dense)
        
        # Convolution neural network
        if self.label_columns is not None:
            conv_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=32,
                                       kernel_size=(self.conv_width, ),
                                       activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=1),])
            
            history = self.compile_and_fit(conv_model, self.conv_window)
            IPython.display.clear_output()
            
            val_performance['Conv'] = conv_model.evaluate(self.conv_window.val)
            performance['Conv'] = conv_model.evaluate(self.conv_window.test, verbose=0)
            logging.info("Convolution neural network model on `wide_conv_window`")
            logging.info(f'Input shape: {self.wide_conv_window.example[0].shape}')
            logging.info(f'Output shape: {conv_model(self.wide_conv_window.example[0]).shape}')
            self.wide_conv_window.plot(conv_model)
        
        # Recurrent neural network
        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, feature] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, feature]
            tf.keras.layers.Dense(units=1)])
        
        history = self.compile_and_fit(lstm_model, self.wide_window)
        
        IPython.display.clear_output()
        val_performance['LSTM'] = lstm_model.evaluate(self.wide_window.val)
        performance['LSTM'] = lstm_model.evaluate(self.wide_window.test, verbose=0)
        if self.label_columns is not None:
            logging.info(f'Input shape: {self.wide_window.example[0].shape}')
            logging.info(f'Output shape: {lstm_model(self.wide_window.example[0]).shape}')
            self.wide_window.plot(lstm_model)
        
        # Advanced: Residual connections
        residual_lstm = ResidualWrapper(
            tf.keras.Sequential([
                tf.keras.layers.LSTM(32, return_sequences=True),
                tf.keras.layers.Dense(
                    self.num_features,
                    # The predicted deltas should start small
                    # So initialize the output layer with zeros
                    kernel_initializer=tf.initializers.zeros())]))
        
        history = self.compile_and_fit(residual_lstm, self.wide_window)
        IPython.display.clear_output()
        
        val_performance['Residual LSTM'] = residual_lstm.evaluate(self.wide_window.val)
        performance['Residual LSTM'] = residual_lstm.evaluate(self.wide_window.test, verbose=0)
        
        # Performance
        x = np.arange(len(performance))
        width = 0.3
        metric_index = lstm_model.metrics_names.index('mean_absolute_error')
        val_mea = [v[metric_index] for v in val_performance.values()]
        test_mea = [v[metric_index] for v in performance.values()]
        
        plt.ylabel('mean_absolute_error [T (degC), normalized]')
        plt.bar(x - 0.17, val_mea, width, label='Validation')
        plt.bar(x + 0.17, test_mea, width, label='Test')
        plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
        _ = plt.legend()
        plt.show()
        
        for name, value in performance.items():
            logging.info(f'{name:12s}: {value[1]:0.4f}')