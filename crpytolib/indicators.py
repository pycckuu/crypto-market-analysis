import theano
from theano import tensor
from theano import function
import numpy as np
import pandas as pd


class Indicators:

    class Ichimoku:
        '''Ichimoku Kinko Hyo'''

        def __init__(self, high, low, close):
            self.high_prices = high.values
            self.low_prices = low.values
            self.close_prices = close.values

        def tenkan(self):
            # Tenkan-sen
            # The average of the nine period high and the nine period low

            nine_period_high = pd.rolling_max(self.high_prices, window=9)
            nine_period_low = pd.rolling_min(self.low_prices, window=9)

            nph = tensor.dvector('nph')
            npl = tensor.dvector('npl')
            t = (nph + npl) / 2
            tenkan_func = function([nph, npl], t)
            tenkan = tenkan_func(nine_period_high, nine_period_low)
            return pd.Series(tenkan)

        def kijun(self):
            # Kijun-sen
            # The average of the twenty six period high and the twenty six period low

            twenty_six_period_high = pd.rolling_max(self.high_prices, window=26)
            twenty_six_period_low = pd.rolling_min(self.low_prices, window=26)

            tsph = tensor.dvector('tsph')
            tspl = tensor.dvector('tspl')
            k = (tsph + tspl) / 2
            kijun_func = function([tsph, tspl], k)
            return pd.Series(
                kijun_func(twenty_six_period_high, twenty_six_period_low))

        def chikou(self):
            # Chikou Span
            # Closing price shifted back twenty six periods

            return pd.Series(self.close_prices).shift(-26)

        def senkouA(self):
            # Senkou Span A
            # The average of the Tenkan-sen and the Kijun-sen shifted back twenty six periods

            tenkanS = pd.Series(self.tenkan()).shift(-26)
            kijunS = pd.Series(self.kijun()).shift(-26)
            t = tensor.dvector('t')
            k = tensor.dvector('k')

            sa = ((t + k) / 2)
            senA = function([t, k], sa)

            return pd.Series(senA(tenkanS, kijunS))

        def senkouB(self):
            # Senkou Span B
            # The average of the fifty two period high and the fifty two period low

            fifty_two_period_high = pd.rolling_max(self.high_prices, window=52)
            fifty_two_period_low = pd.rolling_min(self.low_prices, window=52)

            ftph = tensor.dvector('ftph')
            ftpl = tensor.dvector('ftpl')

            sb = (ftph + ftpl) / 2
            senB = function([ftph, ftpl], sb)

            return pd.Series(senB(fifty_two_period_high,
                                  fifty_two_period_low)).shift(26)

    class Macd:
        '''Moving Average Convergence Divergence - Note: center of mass(com) is the (period-1)/2'''

        def __init__(self, close):
            self.close_prices = close.values

        def twenty_six_ema(self):
            # Twenty Six Period Exponential Moving Average

            tspema = pd.Series(
                pd.ewma(self.close_prices, com=12.5, min_periods=26))

            return tspema

        def twelve_ema(self):
            # Twelve Period Exponential Moving Average

            tpema = pd.Series(
                pd.ewma(self.close_prices, com=5.5, min_periods=12))

            return tpema

        def diff_ema(self):
            # Difference of the two Exponential Moving Averages

            x_tpema = tensor.dvector('x_tpema')
            y_tspema = tensor.dvector('y_tspema')
            s = y_tspema - x_tpema
            diff_func = function([x_tpema, y_tspema], s)
            difference = diff_func(self.twelve_ema(), self.twenty_six_ema())

            return pd.Series(difference)

        def nine_ema_of_diff(self):
            # Nine Period Moving Average of Difference

            npema = pd.ewma(self.diff_ema(), com=4, min_periods=9)

            return pd.Series(npema)

    class BollingerBand:
        '''Bollinger Bands (Used to measure volatility)'''

        def __init__(self, close):
            self.close_prices = close.values

            # Twenty period simple moving average of close price
            self.twentySMA = pd.rolling_mean(self.close_prices, window=20)
            # Twenty period standard deviation of close price
            self.twentySD = pd.rolling_std(self.close_prices, window=20)

            tsma = tensor.dvector('tsma')
            tsd = tensor.dvector('tsd')

            m = tsma
            u = tsma + (tsd * 2)
            l = tsma - (tsd * 2)

            self.middle_func = function([tsma], m)
            self.upper_func = function([tsma, tsd], u)
            self.lower_func = function([tsma, tsd], l)

        def middle(self):
            # Middle Bollinger Band (Twenty period SMA)
            middle_band = self.middle_func(self.twentySMA)
            return pd.Series(middle_band)

        def upper(self):
            # Upper Bollinger Band (Twenty period SMA plus twice the twenty period standard deviation)
            upper_band = self.upper_func(self.twentySMA, self.twentySD)
            return pd.Series(upper_band)

        def lower(self):
            # Lower Bollinger Band (Twenty period SMA minus twice the twenty period standard deviation)
            lower_band = self.lower_func(self.twentySMA, self.twentySD)
            return pd.Series(lower_band)

        def percentB(self):
            # Percent B
            upperb = self.upper().fillna(0)
            lowerb = self.lower().fillna(0)

            return ((pd.Series(self.close_prices).astype('float64') - lowerb) /
                    (upperb - lowerb)).replace([np.inf, -np.inf], np.nan)

    class Obv:
        '''On balance volume'''

        def __init__(self, close, volume):
            self.close_prices = close
            self.volume = volume

            # Structing dataframe where there are two columns ('Close' amd 'Volume') and the datatypes are numeric

            data = pd.concat(
                [pd.DataFrame(self.close_prices),
                 pd.DataFrame(self.volume)],
                axis=1)
            data = data.convert_objects(convert_numeric=True)
            data.columns = ['Close', 'Volume']

            # Creating obv column
            # Add volume if close greater than close-1, subtract if close less than close-1, no change if close equal close-1

            obv = [0] * len(data.index)
            for i in range(len(data.index)):
                if i - 1 == -1:
                    obv[i - 1] = 0
                elif data['Close'].loc[i] > data['Close'].loc[i - 1]:
                    obv[i] = obv[i - 1] + data['Volume'].loc[i]
                elif data['Close'].loc[i] < data['Close'].loc[i - 1]:
                    obv[i] = obv[i - 1] - data['Volume'].loc[i]
                else:
                    obv[i] = obv[i - 1]
            data['OBV'] = obv

        def values(self):
            return pd.Series(data['OBV'])
