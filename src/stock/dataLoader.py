# TODO: load stock data
# read data to file=
import ffn
import pandas as pd
import os


class StockLoader(object):
    def __init__(self, stock, *args):
        super(StockLoader, self).__init__(*args)
        self.stock = stock
        self.df: pd.DataFrame = None

    def LoadStock(self, columns: list, start: str, end: str, dir: str, days: int):
        self.start = start
        self.end = end
        self.dir = dir
        self.days = days
        if not os.path.exists(dir):
            print(f"Creating {dir} .. ..")
            os.makedirs(dir)

        colStr = ','.join([f'{self.stock}:{col}' for col in columns])
        self.df = ffn.get(colStr, start=start, end=end)
        self.GetRateFromPeriod(days=days)
        idx = pd.date_range(start=start, end=end)
        self.df = self.df.reindex(idx).fillna(method="ffill")

    def saveStock(self):
        filename = f'{self.stock}_{self.start.replace("_","")}_{self.end.replace("_","")}_{self.days}.csv'
        fullPath = os.path.join(self.dir, filename)
        print(f'Writing file to {fullPath}')
        self.df.to_csv(fullPath)

    # def GetRateFromPeriod(self, start: str, end: str):
    #     prices = ffn.get(self.stock, start=start, end=end)
    #     return prices.calc_total_return()[0]

    def GetRateFromPeriod(self, days: int):
        self.df["close_rate"] = self.df[f"{self.stock.lower()}close"].shift(-days) - self.df[f"{self.stock.lower()}close"]
        self.df["close_rate"] = self.df["close_rate"] / self.df[f"{self.stock.lower()}close"]

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
        start='2022-05-01', end='2022-05-03'))
