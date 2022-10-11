# Iteration 1

## Solution Setup

The solution of the problem is using AutoARIMA for each Branch and Departement.
We need to iterate the AutoArima for each branch and departement. There is no specific setup for this solutions. 
Make sure for every iteration, the data is sorted by time and the AutoARIMA ready to solve any case like `stationary` and `differentiation`. 

Notes :
* The autotune of AutoARIMA script can be found [here](). 
* The ramal package also in [here]() 

## Summary Results

In this iteration, we have tried several options and here is a comparison (currently, all of the model eval focus on one product):

* model1 --> ARIMA(3,1,2) on `1:92` only.

```
{'MAE': 8421.83,
 'MAPE': 0.0624,
 'MSE': 93343288.83,
 'R2': 0.06935}
```

* model2 --> auto arima without seasonal effect on `1:92` only.
```
{'MAE': 11895.84,
 'MAPE': 0.09049,
 'MSE': 207816530.36,
 'RMSE': 14415.84,
 'R2': -1.0719}
```

* model3 --> ARIMA(3,1,2) x (1, 1, 1, 5) with seasonal on `1:92` only.

```
{'MAE': 8545.41,
 'MAPE': 0.0634,
 'MSE': 94055884.46,
 'R2': 0.0622}
```

* model4 --> auto arima with seasonal effect on `1:92` only. final model is ARIMA(4,1,0)(0,1,1)[5]

```
{'MAE': 7723.93,
 'MAPE': 0.0559,
 'MSE': 66039531.31,
 'R2': 0.3415}
```
