# TODO: load stock data
# read data to file
import ffn
import pandas as pd
import os


class StockLoader(object):
    def __init__(self, stock, *args):
        super(StockLoader, self).__init__(*args)
        self.stock = stock
        self.df = None

    def LoadStock(self, columns: list, start: str, end: str, dir: str):
        filename = f'{self.stock}_{start.replace("_","")}_{end.replace("_","")}.csv'
        fullPath = os.path.join(dir, filename)
        if not os.path.exists(dir):
            print(f"Creating {dir} .. ..")
            os.makedirs(dir)

        colStr = ','.join([f'{self.stock}:{col}' for col in columns])
        self.df = ffn.get(colStr, start=start, end=end)
        print(f'Writing file to {fullPath}')
        self.df.to_csv(fullPath)

    def GetRateFromPeriod(self, start: str, end: str):
        prices = ffn.get(self.stock, start=start, end=end)
        return prices.calc_total_return()[0]

    def PrintStat(self):
        stats = self.df.calc_stats()
        return stats.display()


if __name__ == '__main__':
    stockLoader = StockLoader('TSLA')
    stockLoader.LoadStock(columns=['Open', 'High', 'Low',
                                   'Close'],
                          start='2020-08-21', end='2020-09-02',
                          dir='./data/TSLA_2020_2022/stock_data/')
    print('Many cool stats: ')
    stockLoader.PrintStat()
    print('Stock shape: ', stockLoader.df.shape)
    print('Stock Columns: ', list(stockLoader.df.columns))
    print('Get return in a period: ', stockLoader.GetRateFromPeriod(
        start='2012-01-01', end='2022-05-01'))
