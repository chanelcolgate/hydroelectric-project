import pandas as pd

class GenerateData():
    excelFile = '../data/hydrological.xlsx'
    def __init__(self, typeExcel: int = 0, excelFile: str = 'Not Set'):
        self.typeExcel = typeExcel
        if self.typeExcel == 0:
            self.excelFile = '../data/hydrological.xlsx'
        elif self.typeExcel == 1:
            self.excelFile = '../data/SMP 2018_original.xlsx'
        elif self.excelFile == 2:
            self.excelFile = '.../data/SMP 2019 to 2021.xlsx'
        else:
            self.excelFile = 'Not Set'
            
    def readExcelFile(self):
        with pd.ExcelFile(self.excelFile) as f:
            df1 = pd.read_excel(f, "2018", skiprows=3)
            
        print(df1)
        
def main():
    gd = GenerateData()
    gd.readExcelFile()
    
if __name__ == "__main__":
    main()
        