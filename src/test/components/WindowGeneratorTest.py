import unittest
import logging
import pandas as pd
import numpy as np
import pprint
import tensorflow as tf
from components.WindowGenerator import WindowGenerator

class WindowGeneratorTest(unittest.TestCase):
    INPUT_WIDTH = 6
    LABEL_WIDTH = 1
    SHIFT       = 1
    LABEL_COLUMNS = ['T (degC)']
    EXAMPLE_WINDOW_SHAPE = (3, 7, 19)
    EXAMPLE_INPUTS_SHAPE = (3, 6, 19)
    EXAMPLE_LABELS_SHAPE = (3, 1, 1)
    
    @classmethod
    def setUpClass(self):
        logging.basicConfig(format="%(asctime)s:%(module)s:%(levelname)s:%(message)s", level=logging.DEBUG)
        logging.info("Testing WindowGenerator Class...")
        csv_path = '../../../data/jena_climate_2009_2016.csv'
        df = pd.read_csv(csv_path)
        logging.info('\n' + pprint.pformat(df.head(1)))
        # Wind velocity
        wv = df['wv (m/s)']
        bad_wv = wv == -9999.0
        wv[bad_wv] = 0.0
        
        max_wv = df['max. wv (m/s)']
        bad_max_wv = max_wv == -9999.0
        max_wv[bad_max_wv] = 0.0
        
        # The above inplace edits are reflected in the DataFrame
        # logging.info(pprint.pformat(df['wv (m/s)'].min()))
        
        wv = df.pop('wv (m/s)')
        max_wv = df.pop('max. wv (m/s)')
        
        # Convert to radians
        wd_rad = df.pop('wd (deg)') * np.pi / 180
        
        # Calculate the wind x and y components.
        df['Wx'] = wv*np.cos(wd_rad)
        df['Wy'] = wv*np.sin(wd_rad)
        
        # Calculate the max wind x and y components.
        df['max Wx'] = max_wv*np.cos(wd_rad)
        df['max Wy'] = max_wv*np.sin(wd_rad)
        
        # Time
        date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
        timestamp_s = date_time.map(pd.Timestamp.timestamp)
        
        day = 24*60
        year = (365.2425)*day
        
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi /year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        
        column_indices = {name: i for i, name in enumerate(df.columns)}
        
        n = len(df)
        train_df = df[0:int(n*0.7)]
        val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]
        
        num_features = df.shape[1]
        
        # Normalize the data
        train_mean = train_df.mean()
        train_std = train_df.std()
        
        self.train_df = (train_df - train_mean) / train_std
        self.val_df = (val_df - train_mean) / train_std
        self.test_df = (test_df - train_mean) / train_std
        
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testDataWindowing(self):
        w1 = self._createWindowGenerator()
        logging.info('\n' + str(w1))
        # w2 = WindowGenerator(input_width=24,
        #                      label_width=1,
        #                      shift=24,
        #                      train_df=self.train_df,
        #                      val_df=self.val_df,
        #                      test_df=self.test_df,
        #                      label_columns=['T (degC)'])
        # logging.info('\n' + str(w2))
        
    def testSplitWindow(self):
        w2 = self._createWindowGenerator()
        
        # Stack three slices, the length of the length of the total window:
        example_window = tf.stack([np.array(self.train_df[:w2.total_window_size]),
                                   np.array(self.train_df[100:100+w2.total_window_size]),
                                   np.array(self.train_df[200:200+w2.total_window_size])])
        
        example_inputs, example_labels = w2.split_window(example_window)
        
        self.assertEqual(example_window.shape, self.EXAMPLE_WINDOW_SHAPE)
        self.assertEqual(example_inputs.shape, self.EXAMPLE_INPUTS_SHAPE)
        self.assertEqual(example_labels.shape, self.EXAMPLE_LABELS_SHAPE)
        w2.plot()
        logging.info('All shapes are: (batch, time, features)')
        logging.info(f'Window shape: {example_window.shape}')
        logging.info(f'Inputs shape: {example_inputs.shape}')
        logging.info(f'labels shape: {example_labels.shape}')
        
        
    def testMakeDataset(self):
        w2 = self._createWindowGenerator()
        # w2.train.element_spec
        w2.plot()
        # w2.plot(plot_col='p (mbar)')
        for example_inputs, example_labels in w2.train.take(1):
            logging.info('All shapes are: (batch, time, features)')
            logging.info(f'Inputs shape: {example_inputs.shape}')
            logging.info(f'labels shape: {example_labels.shape}')
    
    def _createWindowGenerator(self):
        return WindowGenerator(input_width=self.INPUT_WIDTH, label_width=self.LABEL_WIDTH, shift=self.SHIFT,
                               train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
                               label_columns=self.LABEL_COLUMNS)
        
if __name__ == "__main__":
    unittest.main()