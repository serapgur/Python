##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction-LCW
##############################################################


import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Proje/Studies/Studies w_Datav2/CLTV/CLTV_LCW.csv")
df.head()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


##############################################################
# 1. EDA
##############################################################

# Kategorik ve numerik değişkenleri ayıralım.
def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Numerik değişkenleri kontrol edip yanlış atamaları düzeltelim.
df[num_cols].head()
num_cols = [i for i in num_cols if i not in ["Musteriref", "UrunRef", "OrderId", "OrderDeliveryCityRef", "OrderDeliveryCountyRef","SiparisYilAyGun","FaturaYilAyGun","RFM_SCORE","Beden_Number_Part"]]

# Kategorik değişkenleri kontrol edip yanlış atamaları düzeltelim.
df[cat_cols].head()

# Numerik değişkenlerden çıkardığımız bazı değişkenleri kategorik değişken olarak ekleyelim.

Beden_Number_Part = cat_cols + ["OrderDeliveryCityRef","RFM_SCORE","Beden_Number_Part"]


# Aykırı değer incelemesi yapalım.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

# Aykırı değerleri görselleştirelim.

for col in num_cols:
    sns.boxplot(x=df[col])
    plt.show()

# Aykırı değerleri silelim.

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

for col in num_cols:
    df2=remove_outlier(df, col)

# Ne kadar gözlem kaybettik bakalım.

df.shape[0]-df2.shape[0] #2403 gözlem kaybettik.
(df.shape[0]-df2.shape[0])/(df.shape[0])*100 # %2.369'unu sildik.

df = df2.copy()

# Eksik gözlem analizi yapalım.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Eksik gözlemimiz bulunmuyor.

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (daha önce cltv_c'de analiz gününe göre, burada kullanıcı özelinde)
# T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

df["OrderDate"] = df["OrderDate"].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
today_date = df["OrderDate"].max()+timedelta(days=1)

cltv_df = df.groupby('MusteriRef').agg({'OrderDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'OrderId': lambda num: num.nunique(),
                                         'UnitPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency_cltv', 'T', 'frequency_cltv', 'monetary_cltv']
cltv_df.head()


# Müşteri ID'lerine bağlı olarak tenure verilerini esas veri setimize ekleyelim.

tenure = cltv_df["T"]
df_merge = pd.merge(df, tenure, how="left", left_on="MusteriRef", right_on="MusteriRef")
df_merge.head()

# Tenure eklenen halini kayıt edelim.

path = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\TENURE\\Tenure_LCW.csv"
df_merge.to_csv(path, index_label=False,index=False)








