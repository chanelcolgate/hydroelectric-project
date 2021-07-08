import logging
import unittest
from prepare.GenerateData import GenerateData

class GenerateDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        logging.basicConfig(format='%(asctime)s:%(module)s:%(levelname)s:%(message)s', level=logging.DEBUG)
        logging.info('Testing GenerateData Class...')
        self.generateData = GenerateData()
        
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testReadExcelFile(self):
        df = self.generateData.readExcelFile()
        self.assertEqual(df.shape[0], 1185)
        logging.info(df.head())
        
if __name__ == '__main__':
    unittest.main()