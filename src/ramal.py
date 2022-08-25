import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer,r2_score, mean_squared_log_error, mean_absolute_percentage_error


def plot_per_product(df,pr_id,data_col = 'qu',kind = 'line'):
    df_temp = df[df['ids'] == pr_id]
    df_temp.set_index('Date', inplace = True)
    df_temp[data_col].plot(kind = kind, title = pr_id)


class jampi():
    def __init__(self, data, time_col, data_col, freq=None, split=None, start_test=None, s_factor=1):
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.set_index(time_col)
        self.data_col = data_col 
        self.time_col = time_col
        self.seasonal_factor = s_factor
        
        if freq:
            self.freq = freq
            self.data = data.asfreq(self.freq)[[self.data_col]]
        else:
            self.data = data[[self.data_col]]
        
        
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
                
    def data_diff(self,differencing=0, seasonal_differencing=0, s_factor=7):
        if differencing==0:
            dat_ts = self.data
        else:
            dat_ts = self.data.diff(differencing).dropna()
        
        if seasonal_differencing>0:
            dat_ts = dat_ts.diff(seasonal_differencing*s_factor).dropna()
        
        return dat_ts
    
    def data_1d(self,test=False):
        if test :
            return self.data_test[self.data_col].values
        else:
            return self.data[self.data_col].values
            
            
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
        
        plt.legend()
            
    def decompose(self, differencing=0, seasonal_differencing=0):
        # sd1 = seasonal_decompose(x = ts.data, model = 'multiplicative')
        dat_ts = self.data_diff(differencing, seasonal_differencing, self.seasonal_factor)
        self.decomp = seasonal_decompose(x = dat_ts, model = 'additive')
        
    def adf_test(self, differencing=0, seasonal_differencing=0):
        dat_ts = self.data_diff(differencing, seasonal_differencing, self.seasonal_factor)
        result = adfuller(dat_ts[self.data_col].values, autolag='AIC')
        
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
        
    def acf_pacf(self, logs, differencing=0, seasonal_differencing=0):
        dat_ts = self.data_diff(differencing, seasonal_differencing, self.seasonal_factor)
        
        fig, axes = plt.subplots(2,1,figsize=(12,8))
        plot_acf(dat_ts, lags=logs, ax=axes[0], zero=False, auto_ylims=True)
        plot_pacf(dat_ts, lags=logs, ax=axes[1], zero=False, auto_ylims=True)
        
    def auto_arma_param(self, differencing=0, seasonal_differencing=0):
        dat_ts = self.data_diff(differencing, seasonal_differencing, self.seasonal_factor)
        
        self.acf_val, acf_ci = acf(dat_ts, alpha=0.05,nlags = 7)
        self.acf_ci = np.array([x-y for x,y in zip(self.acf_val, acf_ci)])
        self.acf_diff = np.array([1 if abs(x)-y[0]>0 else 0 for x,y in zip(self.acf_val, self.acf_ci)])        
        
        self.pacf_val, pacf_ci = pacf(dat_ts, alpha=0.05,nlags = 7)
        self.pacf_ci = np.array([x-y for x,y in zip(self.pacf_val, pacf_ci)])
        self.pacf_diff = np.array([1 if abs(x)-y[0]>0 else 0 for x,y in zip(self.pacf_val, self.pacf_ci)])
        
        ma_param = np.argmin(self.acf_diff)-1
        ar_param = np.argmin(self.pacf_diff)-1
        
        
        print("MA("+str(ma_param)+")")
        print("AR("+str(ar_param)+")")
        if differencing>0:
            print("with ",differencing,"differencing")
        
        

def eval_model(ytrue,ypred,thrs=0):
    all_pair = np.array(list(zip(ytrue,ypred)))
    overpred_pair = all_pair[all_pair[:,0]<all_pair[:,1]]
    underpred_pair = all_pair[all_pair[:,0]>all_pair[:,1]]
    all_err=np.abs(ypred-ytrue)
    summary_eval = {}
    
    summary_eval['MAE']=all_err.mean()
    summary_eval['MAPE']=mean_absolute_percentage_error(ytrue,ypred)
    summary_eval['MSE']=mean_squared_error(ytrue,ypred)
    summary_eval['over_est_'+str(thrs)]=(ypred-ytrue>thrs).mean()
    summary_eval['under_est_'+str(thrs)]=(ytrue-ypred>thrs).mean()
    summary_eval['R2']=r2_score(ytrue,ypred)
    # summary_eval['RMSLE']=np.sqrt(mean_squared_log_error(ytrue,ypred))
    
    return summary_eval

        
    
