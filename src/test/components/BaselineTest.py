import logging
import unittest
import pandas as pd
import numpy as np
import pprint
import tensorflow as tf
from components.Baseline import Baseline
from components.WindowGenerator import WindowGenerator

class BaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        logging.basicConfig(format="%(asctime)s:%(module)s:%(levelname)s:%(message)s", level=logging.DEBUG)
        logging.info("Testing BaselineTest Class...")
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
        
        self.column_indices = {name: i for i, name in enumerate(df.columns)}
        
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
    
    def testCall(self):
        single_step_window = WindowGenerator(input_width=1,
                             label_width=1,
                             shift=1,
                             train_df=self.train_df,
                             val_df=self.val_df,
                             test_df=self.test_df,
                             label_columns=['T (degC)'])
        # logging.info('\n' + str(single_step_window))
        baseline = Baseline(label_index = self.column_indices['T (degC)'])
        
        baseline.compile(loss=tf.losses.MeanSquaredError(),
                         metrics=[tf.metrics.MeanAbsoluteError()])
        
        val_performance = {}
        performance = {}
        val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
        performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
    
if __name__ == '__main__':
    unittest.main()