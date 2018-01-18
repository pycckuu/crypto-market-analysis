import pandas as pd


class Coin:
    """class containing coin prices dataframe
    """

    def __init__(self, high, low, close):
        self.high_prices = high.values
        self.low_prices = low.values
        self.close_prices = close.values
