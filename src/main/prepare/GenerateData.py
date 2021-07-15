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
            if self.excelFile == '../../../data/SMP 2018_original.xlsx':
                for sheet in f.sheet_names:
                    if sheet == 'SMP':
                        # Create column names
                        names = ['date']
                        names.extend([str(i) for i in range(25) if i > 0])
                        part = pd.read_excel(f, sheet, skiprows=1, usecols="B:Z",
                                             names=names)
                        # Transform column `date` from `datetime` to type `str`
                        part.loc[:, 'date'] = part['date'].astype(str)
                        # Create  DataFrame contains 8640 rows and columns are `date`, `time` and `value`
                        daily_index = pd.date_range("2018-01-01", "2018-12-31", freq="D").astype('str').to_list()
                        index = pd.MultiIndex.from_product([daily_index, [i for i in range(25) if i > 0]], names=["date", "time"])
                        df = pd.DataFrame(0, index=index, columns=["value"])
                        df = df.reset_index()
                        df['date time'] = df.loc[:, 'date'].astype(str) + ' ' + df.loc[:, 'time'].astype(str)
                        for column in [str(i) for i in range(25) if i > 0]:
                            for i in range(part.shape[0]):
                                dt = df['date time'] == part.loc[i, 'date'] + ' ' + column
                                df['value'][dt] = part.loc[i, column]
                        return df
    
    def createDataFrame(self, dirFile):
        df = self.readExcelFile()
        df.to_csv(dirFile)
        return True
        
def main():
    gd = GenerateData()
    gd.readExcelFile()
    
if __name__ == "__main__":
    main()
        