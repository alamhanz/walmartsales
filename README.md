# walmartsales

Its a timeseries usecase

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


