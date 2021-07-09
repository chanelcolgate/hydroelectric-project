import pandas as pd

class GenerateData():
    def __init__(self, typeExcel: int = 0):
        self.typeExcel = typeExcel
        if self.typeExcel == 0:
            self.excelFile = '../../../data/hydrological.xlsx'
        elif self.typeExcel == 1:
            self.excelFile = '../../../data/SMP 2018_original.xlsx'
        elif self.excelFile == 2:
            self.excelFile = '../../../data/SMP 2019 to 2021.xlsx'
        else:
            self.excelFile = 'Not Set'
            
    def readExcelFile(self):
        with pd.ExcelFile(self.excelFile) as f:
            if self.excelFile == '../../../data/hydrological.xlsx':
                parts = []
                for sheet in f.sheet_names:
                    part = pd.read_excel(f, sheet, skiprows=0, usecols="B:I",
                                        names=['week', 'day', 'month', 'height lake', 
                                                'height downstream', 'flow lake',
                                                'flow flood', 'flow downstream'])
                    part['time'] = sheet + '-' + part.loc[:, 'month'].astype(str) + '-' + part.loc[:, 'day'].astype(str)+ ' 00:00:00'
                    parts.append(part)
        return pd.concat(parts)
    
    def createDataFrame(self, dirFile):
        df = self.readExcelFile()
        df.to_csv(dirFile)
        return True
        
def main():
    gd = GenerateData()
    gd.readExcelFile()
    
if __name__ == "__main__":
    main()
        