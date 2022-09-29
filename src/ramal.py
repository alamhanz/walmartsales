import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer,r2_score, mean_squared_log_error, mean_absolute_percentage_error
import pmdarima

def min_date_use(ts_data):
    XX = ts_data.cumsum()
    zero_XX = XX[XX['demand']==0]
    if len(zero_XX)>0 :
        min_non_zero_date = max(XX[XX['demand']==0].index)
    else:
        min_non_zero_date = -1
    
    return min_non_zero_date


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
                self.data.drop([start_test], axis=0,inplace =True)
                
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
        
    def adf_test(self, differencing=0, seasonal_differencing=0, verbose = True):
        dat_ts = self.data_diff(differencing, seasonal_differencing, self.seasonal_factor)
        result = adfuller(dat_ts[self.data_col].values, autolag='AIC')

        if result[1]<0.05:
            is_stationary = True
        else:
            is_stationary = False
            
            
        if verbose:
            print(f'ADF Statistic: {result[0]}')
            print(f'p-value: {result[1]}')

            if is_stationary:
                print('Stationary')
            else:
                print('Not Stationary')
        
        return is_stationary

        # KPSS Test
        # result = kpss(ts.data[ts.data_col].values, regression='c')
        # print('\nKPSS Statistic: %f' % result[0])
        # print('p-value: %f' % result[1])
        
    def acf_pacf(self, logs, differencing=0, seasonal_differencing=0):
        dat_ts = self.data_diff(differencing, seasonal_differencing, self.seasonal_factor)
        
        fig, axes = plt.subplots(2,1,figsize=(12,8))
        plot_acf(dat_ts, lags=logs, ax=axes[0], zero=False)
        plot_pacf(dat_ts, lags=logs, ax=axes[1], zero=False)
        
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
    summary_eval['RMSE']=np.sqrt(summary_eval['MSE'])
    summary_eval['over_est_'+str(thrs)]=(ypred-ytrue>thrs).mean()
    summary_eval['under_est_'+str(thrs)]=(ytrue-ypred>thrs).mean()
    summary_eval['R2']=r2_score(ytrue,ypred)
    
    return summary_eval



## Smooth Moving Average

def sma_pred(N, df_ts, col_name):
    df_ts['pred_ts'] = df_ts[col_name].rolling(N, closed='left').mean().fillna(0).astype(int)
    return df_ts
    

def sma_eval(N, df_ts, col_name, start_test, eval_metrics = 'MSE', return_pred=False):
    df_ts = sma_pred(N, df_ts, col_name)

    train_eval = df_ts[N:][:start_test]
    train_eval.drop([start_test], axis=0,inplace =True)
    test_eval = df_ts[start_test:]

    train_ev = eval_model(train_eval[col_name], train_eval['pred_ts'])
    test_ev = eval_model(test_eval[col_name], test_eval['pred_ts'])
    
    if return_pred:
        return train_ev[eval_metrics], test_ev[eval_metrics], df_ts
    else:
        return train_ev[eval_metrics], test_ev[eval_metrics]


def tuning_sma(df_ts, col_name, start_test, eval_metrics = 'MSE', n_rolling=10, opt_method='test-only'):
    
    min_non_zero = min_date_use(df_ts)
    if min_non_zero != -1:
        df_ts = df_ts[min_non_zero:]
    
    all_errors = []
    for i in range(n_rolling):
        err_temp = sma_eval(i+1, df_ts, 'demand', start_test, eval_metrics = eval_metrics)
        all_errors.append(err_temp)
    
    df_error = pd.DataFrame(all_errors, columns = ['train_err','test_err'])
    
    if opt_method=='test-only':
        return df_error, df_error['test_err'].idxmin()+1
    elif opt_method=='test-only2':
        df_temp = df_error[1:].copy()
        return df_error, df_temp['test_err'].idxmin()+1
    else:
        n_opt_train = df_error['train_err'].idxmin()+1
        df_temp = df_error[1:n_opt_train].copy()
        return df_error, df_temp['test_err'].idxmin()+1


## Auto tune Arima
def auto_tune_arima(df_ts,start_test,time_col='MonthYear',data_col='demand', random_fits=30):
    ts = jampi(data = df_ts, time_col = time_col, data_col = data_col, start_test = start_test)
    
    min_non_zero = min_date_use(df_ts)
    if min_non_zero != -1:
        df_ts = df_ts[min_non_zero:]
    
    is_stationary = ts.adf_test(verbose = False)

    data_to_fit = pd.concat([ts.data,ts.data_test])
    oob_data = len(ts.data_test)

    ## non seasonal
    curr_mae = 10000000
    curr_model1 = '-'
    for diff_cnt in range(2):
        ts_model1 = pmdarima.arima.auto_arima(data_to_fit, d = diff_cnt, max_d = 2
                                 ,start_p = 0, max_p = 4
                                 ,start_q = 0, max_q = 4
                                 ,seasonal = False
                                 ,stationary = is_stationary
                                 ,stepwise = False
                                 ,n_jobs = 5
                                 ,random = True
                                 ,out_of_sample_size=oob_data
                                 ,information_criterion='oob'
                                 ,scoring = 'mae'
                                 ,n_fits = random_fits)

        if ts_model1.oob() <= curr_mae:
            curr_model1 = ts_model1
            curr_mae = ts_model1.oob()

    ## seasonal 
    curr_mae = 10000000
    curr_model2 = '-'
    # diff_cnt = 0 ## need to have 1 and 2
    # diff_s_cnt = 0 ## need to have 1 and 2
    # m_season = 4 ## need to have 2 to 6
    for m_season in range(2,7):
        for diff_s_cnt in range(2):
            for diff_cnt in range(2):
                try:
                    ts_model2 = pmdarima.arima.auto_arima(data_to_fit, d = diff_cnt, max_d = 2
                                             ,start_p = 0, max_p = 4
                                             ,start_q = 0, max_q = 4
                                             ,D = diff_s_cnt, max_D = 2
                                             ,start_P = 0, max_P = 4
                                             ,start_Q = 0, max_Q = 4
                                             ,m = m_season
                                             ,seasonal = True
                                             ,stationary = is_stationary
                                             ,stepwise = False
                                             ,n_jobs = 5
                                             ,random = True
                                             ,out_of_sample_size=oob_data
                                             ,information_criterion='oob'
                                             ,scoring = 'mae'
                                             ,n_fits = random_fits)
                    if ts_model2.oob() <= curr_mae:
                        curr_model2 = ts_model2
                        curr_mae = ts_model2.oob()
                except:
                    print('Error in : ',(diff_cnt,diff_s_cnt,m_season))
                    pass
                    

    if curr_model2.oob() <= curr_model1.oob():
        ts_model_temp = curr_model2
    else:
        ts_model_temp = curr_model1
        
    return ts_model_temp



def month_with_demand(x):
    x0 = [a for a in x if a>0]
    return len(x0)

def mean_with_demand(x):
    x0 = [a for a in x if a>0]
    return np.mean(x0)

def var_with_demand(x):
    x0 = [a for a in x if a>0]
    if len(x0)>=2:
        return np.var(x0)
    else:
        return np.NaN

def p50_with_demand(x):
    x0 = [a for a in x if a>0]
    if len(x0)>=1:
        return np.quantile(x0,q=0.5)
    else:
        return np.NaN
    
def max_with_demand(x):
    x0 = [a for a in x if a>0]
    if len(x0)>=1:
        return np.max(x0)
    else:
        return np.NaN
    
def min_with_demand(x):
    x0 = [a for a in x if a>0]
    if len(x0)>=1:
        return np.min(x0)
    else:
        return np.NaN
    
def zero_pct(x):
    x0 = [a for a in x if a==0]
    return len(x0)/len(x)

def std_all_demand(x):
    return np.std(x)