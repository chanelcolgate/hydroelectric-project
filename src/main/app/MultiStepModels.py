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

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)
        
    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
    
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)
        
        # Insert the first prediction
        predictions.append(prediction)
        
        # Run the test of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction
            prediciton = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)
            
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, out_steps):
        super().__init__()
        self.out_steps = out_steps
    
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.out_steps, 1])

class RepeatBaseLine(tf.keras.Model):
    def call(self, inputs):
        return inputs

class MultiStepModels():
    MAX_EPOCHS = 1
    def __init__(self, out_steps, num_features, conv_width, multi_window, plot_col):
        self.out_steps = out_steps
        self.multi_window = multi_window
        self.conv_width = conv_width
        self.num_features = num_features
        self.plot_col = plot_col
        
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
        # Baselines
        last_baseline = MultiStepLastBaseline(self.out_steps)
        last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                              metrics=[tf.metrics.MeanAbsoluteError()])
        
        multi_val_performance = {}
        multi_performance = {}
        
        multi_val_performance['Last'] = last_baseline.evaluate(self.multi_window.val)
        multi_performance['Last'] = last_baseline.evaluate(self.multi_window.test, verbose=0)
        self.multi_window.plot(last_baseline)
        
        # Repeat baselines
        repeat_baseline = RepeatBaseLine()
        repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                                metrics=[tf.metrics.MeanAbsoluteError()])
        
        multi_val_performance['Repeat'] = repeat_baseline.evaluate(self.multi_window.val)
        multi_performance['Repeat'] = repeat_baseline.evaluate(self.multi_window.test, verbose=0)
        self.multi_window.plot(repeat_baseline, plot_col=self.plot_col)
        
        # Linear
        multi_linear_model = tf.keras.Sequential([
            #Take the last time-step.
            # Shape [batch, time, feature] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(self.out_steps*self.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.out_steps, self.num_features])])
        
        history = self.compile_and_fit(multi_linear_model, self.multi_window)
        IPython.display.clear_output()
        
        multi_val_performance['Linear'] = multi_linear_model.evaluate(self.multi_window.val)
        multi_performance['Linear'] = multi_linear_model.evaluate(self.multi_window.test, verbose=0)
        self.multi_window.plot(multi_linear_model, plot_col=self.plot_col)
        
        # CNN
        multi_conv_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[: , -self.conv_width:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(self.conv_width)),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(self.out_steps*self.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.out_steps, self.num_features])])
        
        history = self.compile_and_fit(multi_conv_model, self.multi_window)
        IPython.display.clear_output()
        
        multi_val_performance['Conv'] = multi_conv_model.evaluate(self.multi_window.val)
        multi_performance['Conv'] = multi_conv_model.evaluate(self.multi_window.test, verbose=0)
        self.multi_window.plot(multi_conv_model, plot_col=self.plot_col)
        
        # RNN
        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(self.out_steps*self.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.out_steps, self.num_features])
            ])
        
        history = self.compile_and_fit(multi_lstm_model, self.multi_window)
        IPython.display.clear_output()
        
        multi_val_performance['LSTM'] = multi_lstm_model.evaluate(self.multi_window.val)
        multi_performance['LSTM'] = multi_lstm_model.evaluate(self.multi_window.test, verbose=0)
        self.multi_window.plot(multi_lstm_model, plot_col=self.plot_col)
        
        # Advanced: Autogressive model
        # RNN
        feedback_model = FeedBack(units=32, out_steps=self.out_steps, num_features=self.num_features)
        history = self.compile_and_fit(feedback_model, self.multi_window)
        IPython.display.clear_output()
        
        multi_val_performance['AR LSTM'] = feedback_model.evaluate(self.multi_window.val)
        multi_performance['AR LSTM'] = feedback_model.evaluate(self.multi_window.test, verbose=0)
        self.multi_window.plot(feedback_model, plot_col=self.plot_col)
        
        # Performance
        x = np.arange(len(multi_performance))
        width = 0.3
        
        metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in multi_val_performance.values()]
        test_mae = [v[metric_index] for v in multi_performance.values()]
        
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()
        plt.show()