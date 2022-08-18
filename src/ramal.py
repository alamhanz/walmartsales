import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

class jampi():
    def __init__(self, data, time_col, data_col, freq = 'd', split=None, start_test=None):
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.set_index(time_col)
        self.data = data.asfreq('d')[[data_col]]
        self.data_col = data_col
        self.time_col = time_col
        
        ## missing value check
        self.null_cnt = self.data[self.data_col].isnull().sum()
        if self.null_cnt:
            print('null data has been found')
            
        if split:
            N = int(len(self.data)*split)
            self.data_test = self.data[-N:]
            self.data = self.data[:-N]
        else:
            if start_test:
                self.data_test = self.data[start_test:]
                self.data = self.data[:start_test]
            
            
    def interpolate(self):
        self.data = self.data.interpolate()
        
    def plot(self,figs=(18,10), test=False):
        if test:
            plt.figure(figsize=figs)
            plt.plot(self.data, label='train', color='steelblue')
            plt.plot(self.data_test, label='test', color='green')
        else:            
            plt.figure(figsize=figs)
            plt.plot(self.data, label='train', color='steelblue')
            
    def decompose(self, differencing=0):
        # sd1 = seasonal_decompose(x = ts.data, model = 'multiplicative')
        if differencing == 0 :
            self.decomp = seasonal_decompose(x = self.data, model = 'additive')
        else:
            self.decomp = seasonal_decompose(x = self.data.diff(differencing).dropna(), model = 'additive')
        
    def adf_test(self, differencing=0):
        if differencing == 0 :
            result = adfuller(self.data[self.data_col].values, autolag='AIC')
        else:
            result = adfuller(self.data.diff(differencing).dropna()[self.data_col].values, autolag='AIC')
            
            
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')

        if result[1]<0.05:
            print('Stationary')
        else:
            print('Not Stationary')

        # KPSS Test
        # result = kpss(ts.data[ts.data_col].values, regression='c')
        # print('\nKPSS Statistic: %f' % result[0])
        # print('p-value: %f' % result[1])
        
    
