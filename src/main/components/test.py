import pandas as pd # type: ignore[import]
from fastapi import FastAPI # type: ignore[import]
from typing import List, Any, Dict

class Report:
    def __init__(
            self,
            priceContract: float,
            data: Dict[str, object]) -> None:
        self.priceContract = priceContract
        df = pd.DataFrame(data = data)
        df['Doanh thu dự kiến TT'] = df['Sản lượng hợp đồng (Qc)']*(priceContract - df['Giá biên tham chiếu cho bản chào giá dự kiến'] - df['Giá CAN']) + \
                                df['Sản lượng dự kiến phát']*(df['Giá biên tham chiếu cho bản chào giá dự kiến'] + df['Giá CAN'])
        df['Doanh thu dự kiến theo HĐ'] = df['Sản lượng dự kiến phát']*priceContract
        self.df = df

    def createReport(self, nameExcel: str = 'Bảng dự kiến chào giá ngày 07/09/2021 với mức công suất 11 kW') -> None:
        with pd.ExcelWriter(f"./file_report/{nameExcel}.xlsx", date_format="dd/mm/yyyy") as writer:
            startrow, startcol = 4, 0
            self.df.to_excel(writer, header=False, index=False,
                             startrow=startrow, startcol=startcol)
            book = writer.book
            sheet = writer.sheets["Sheet1"]
            sheet.write("A2", "Giá HĐ")
            sheet.write("B2", self.priceContract)
            sheet.write("A3:B3", "Ngày 07/09/2021")
            sheet.write("A53:B53", "Tổng doanh thu")
            sheet.write("E53", self.df['Sản lượng hợp đồng (Qc)'].sum())
            sheet.write("F53", self.df['Doanh thu dự kiến TT'].sum())
            sheet.write("G53", self.df['Doanh thu dự kiến theo HĐ'].sum())

            format_header = book.add_format({
                'text_wrap': True,
                'align': 'center',
                'valign':   'vcenter',
                'bold': True
                #'border_color': '#000000',
                #'bottom': 1,
                #'top': 1,
                #'right': 1,
                #'left': 1
            })
            sheet.set_row(3, 80, format_header)
            for colx, value in enumerate(self.df.columns.values):
                sheet.write(3, colx, value)

app = FastAPI()

@app.post("/revenue") # type: ignore[misc]
async def upload_data(
        nameExcel: str = 'Bảng dự kiến chào giá ngày 07-09-2021 với mức công suất 11 kW',
        priceContract: float = 0.0,
        expectedOutput: List[float] = [0.0],
        expectedPrice: List[float] = [0.0],
        priceCan: List[float] = [0.0],
        outputContract: List[float] = [0.0]) -> bool:
    data = {
            'Chu kỳ': [i for i in range(1, 49)],
            'Sản lượng dự kiến phát': expectedOutput,
            'Giá biên tham chiếu cho bản chào giá dự kiến': expectedPrice,
            'Giá CAN': priceCan,
            'Sản lượng hợp đồng (Qc)': outputContract}
    r = Report(priceContract = priceContract, data = data)
    r.createReport(nameExcel = nameExcel)
    return True
