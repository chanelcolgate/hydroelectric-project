import logging
import unittest
import pprint
from prepare.GenerateData import GenerateData

class GenerateDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        logging.basicConfig(format='%(asctime)s:%(module)s:%(levelname)s:\n%(message)s', level=logging.DEBUG)
        logging.info('Testing GenerateData Class...')
        # self.generateData = GenerateData()
        self.generateData = GenerateData(typeExcel=1)
        
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    # def testReadExcelFile(self):
    #     df = self.generateData.readExcelFile()
    #     self.assertEqual(df.shape[0], 1184)
    #     logging.info(df)
        
        
    # def testCreateDataFrame(self):
    #     self.assertTrue(self.generateData.createDataFrame('../../../data/data1.csv'))
        
    # def testReadExcelFile_2(self):
    #     logging.info(f"\n {pprint.pformat(self.generateData.readExcelFile())}")
        
    def testCreateDataFrame(self):
        self.assertTrue(self.generateData.createDataFrame('../../../data/data3.csv'))
    
if __name__ == '__main__':
    unittest.main()