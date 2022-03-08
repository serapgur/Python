###############################################################
# LC Waikiki Datası ile RFM
###############################################################

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_excel("Proje/Studies/Studies w_Datav2/RFM/New_data_LCW.xlsx")

df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Describe #####################")
    print(dataframe.describe().T)

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


# Eksik Gözlem Analizi Yapalım.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

# Eksik değer problemimiz yok.


check_df(df)
"""
Quantity ve linetotal değişkenlerinde negatif değerler yok yani iade durumu verisetinde mevcut değil. Bu sebeple RFM değerlerimizi hesaplayabiliriz. 
"""

df.OrderDate.max()
# Timestamp('2019-12-31 01:07:11.333000')

"""
Veri setindeki son işlem 31 Aralık 2019'da oluşturulmuş. Eğer recency için bugünün tarihini alırsak doğru olmaz, sanki değerlendirmeyi 1 Ocak 2020'de yapıyormuşuz gibi düşünüp analiz tarihini öyle alalım.
"""

today_date = df["OrderDate"].max()+timedelta(days=1)

# RFM metriklerimizi oluşturalım.
rfm = df.groupby('MusteriRef').agg({'OrderDate': lambda OrderDate: (today_date - OrderDate.max()).days,
                                     'OrderId': lambda num: num.nunique(),
                                     'UnitPrice': lambda x: x.sum()})

rfm.columns = ['recency', 'frequency', 'monetary']

# RFM metriklerimizi 1-5 arasına scale edelim.

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])


rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])


rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Recency ve Frequency metriklerimizi birleştirelim.

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm[rfm["RFM_SCORE"] == "55"].head()

# RFM Segmentlerini Oluşturalım

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)


# Sonuçları görselleştirelim.

y = pd.DataFrame(rfm["segment"].value_counts())
plt.figure(figsize=(14,8))
sns.barplot(data=y,x="segment",y=y.index, palette="Greens_d")
plt.show()




# Müşteri ID'lerine bağlı olarak bu rfm verilerini esas veri setimize ekleyelim.

df_merge = pd.merge(df, rfm, how="left", left_on="MusteriRef", right_on="MusteriRef")
df_merge.head()

# Verimizin son halini csv olarak kayıt edelim.

path = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\RFM\\RFM_LCW.csv "
df_merge.to_csv(path,index_label=False,index=False)





