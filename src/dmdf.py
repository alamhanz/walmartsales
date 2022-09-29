import boto3
import io
# demand forecasting tools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_ts_per_product(df,pr_id,col_name='demand', test = None):
    df_temp = df[df['ids'] == pr_id]
    df_temp.set_index('MonthYear', inplace = True)
    if test :
        plt.plot(df_temp[:test][col_name], label='train', color='steelblue')
        plt.plot(df_temp[test:][col_name], label='test', color='green')
        plt.xticks(rotation=90)
    else:
        df_temp[col_name].plot()
        


# inisiasi folder s3 yang akan dibaca
my_bucket = "xxx" # deklarasikan bucket tempat data simpan
dataset = "xxx/" # folder data
s3 = boto3.client("s3") # inisiasi koneksi ke s3 AWS

def gather_data(filename, types='excel', encoding=None, sep=None):
    # gather object
    obj = s3.get_object(Bucket=my_bucket, Key=dataset + filename)
    pseudofile = io.BytesIO(obj['Body'].read())
    if types=='excel':
        df = pd.ExcelFile(pseudofile)
    elif types == 'csv':
        df = pd.read_csv(pseudofile, encoding=encoding, sep=sep)
    return df