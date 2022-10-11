# walmartsales

It is a timeseries usecase. The ultimate problem is to predict walmart sales for the next (at least) one week using previous week data. However, walmart has multiple branch with different departement in each brand.

## Setup

Start the project with environment setup and run the jupyterlab

```
pip install virtualenv
virtualenv myev
source myev/bin/activate
pip install -r requirements.txt
jupyter lab
```
or run this script for windows users
```
pip install virtualenv
virtualenv myev
.\myev\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Structure

```
    |--artifacts
    |--data
        |--raw
        |--interim
        |--processed
        |--externals
    |--notebooks
    |--queries
    |--reports
        |--figures
    |--src
```

## Reference

* Solution iteration 1
* Solution iteration 3
* [dataset source](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
* [Time series guide](https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775)
