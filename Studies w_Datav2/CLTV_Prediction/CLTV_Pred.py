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

df = pd.read_csv("Proje/Studies/Studies w_Datav2/CHURN/Churn_LCW.csv")

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


# Aykırı değerleri Tenure değişkenini oluştururken incelemiştik. Bir daha incelemeyeceğiz çünkü her seferinde yeni aykırı değer çıkacaktır ve veriyi manipüle etmiş oluruz.


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

# Eksik gözlemimiz yok, herhangi bir işlem yapmamıza gerek yok.



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

cltv_df["monetary_cltv"] = cltv_df["monetary_cltv"] / cltv_df["frequency_cltv"]

cltv_df = cltv_df[(cltv_df['frequency_cltv'] > 1)]

cltv_df = cltv_df[cltv_df["monetary_cltv"] > 0]

cltv_df["recency_cltv"] = cltv_df["recency_cltv"] / 7

cltv_df["T"] = cltv_df["T"] / 7

#############################################################################
# 2. BG-NBD Modelinin Kurulması for "Expected Number of Transaction"
#############################################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency_cltv'],
        cltv_df['recency_cltv'],
        cltv_df['T'])

################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency_cltv'],
                                                        cltv_df['recency_cltv'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency_cltv'],
                                              cltv_df['recency_cltv'],
                                              cltv_df['T'])


################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################


bgf.predict(4,
            cltv_df['frequency_cltv'],
            cltv_df['recency_cltv'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency_cltv'],
                                               cltv_df['recency_cltv'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_1_month", ascending=False).head(20)


################################################################
# 1 Ay içinde tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################


bgf.predict(4,
            cltv_df['frequency_cltv'],
            cltv_df['recency_cltv'],
            cltv_df['T']).sum()

# 1 ay içinde şirketin beklenen satış sayısı 1380.51'dir.

################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################


bgf.predict(4 * 3,
            cltv_df['frequency_cltv'],
            cltv_df['recency_cltv'],
            cltv_df['T']).sum()

# 1 ay içinde şirketin beklenen satış sayısı 4100.46'dır.

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

plot_period_transactions(bgf)
plt.show()


##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması for Expected Average Profit
##############################################################


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency_cltv'], cltv_df['monetary_cltv'])

ggf.conditional_expected_average_profit(cltv_df['frequency_cltv'],
                                        cltv_df['monetary_cltv']).head(10)


ggf.conditional_expected_average_profit(cltv_df['frequency_cltv'],
                                        cltv_df['monetary_cltv']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency_cltv'],
                                                                             cltv_df['monetary_cltv'])


##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################



cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency_cltv'],
                                   cltv_df['recency_cltv'],
                                   cltv_df['T'],
                                   cltv_df['monetary_cltv'],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="MusteriRef", how="left")
cltv_final.head()
cltv_final.sort_values(by="clv", ascending=False).head(10)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
cltv_final.sort_values(by="scaled_clv", ascending=False).head()

# 3 aylık süre içinde en çok getirisi olan müşteri 2000 ref numaralı müşteri.

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()


cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)


cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


cltv_final.isnull().sum()


# Ana dosya ile birleştirelim.

del df["T"]
df_merge = pd.merge(df, cltv_final, how="left", left_on="MusteriRef", right_on="MusteriRef")
df_merge.head()
del df_merge["T"]
df_merge.shape # (99048, 54)

df_merge = df_merge.rename(columns={"T":"tenure_cltv_weekly", "recency_cltv":"recency_cltv_weekly", "clv":"clv_3month", "scaled_clv":"scaled_clv_3month", "segment_y":"segment_by_cltv_pred", "segment_x":"segment_by_rfm"})
df_merge.head()



# Asıl veride son 3 aylık toplam harcama ile tahmin edilen 3 aylık clv'yi karşılaştıralım.

df_merge.UnitPrice.sum() # 3274870.725

three_month_ago = df_merge["OrderDate"].max() - timedelta(days=90) # Timestamp('2019-10-02 01:07:11.333000')

df_merge[df_merge["OrderDate"]>three_month_ago]["UnitPrice"].sum()
# 836026.0300000001: Bu değer tüm müşterilerin toplamı fakat cltv pred. yaparken frequency değeri 1'den büyük değerleri almıştık. Sadece o müşterilerin son 3 aylık toplam harcamasına bakalım.

df_merge[(df_merge["OrderDate"]>three_month_ago) & (df_merge["frequency_cltv"]>1) & (df_merge["monetary_cltv"]>0)]["UnitPrice"].sum() # 780354.97 TL

customers = df_merge.groupby("MusteriRef").agg({"clv_3month":"max"})
customers.head()
customers.clv_3month.sum() # 602397.887015783

"""
Son 3 aylık harcama toplamı 836 bin TL iken frequency değeri 1'den fazla olan yani birden fazla alışveriş yapan müşterilerin 3 aylık toplam harcaması 780 bin TL.
3 Aylık cltv tahmini ise 602 bin TL. 
"""



# Veri setinin son halini kayıt edelim.

path_csv = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\CLTV_Prediction\\CLTV_Pred_LCW.csv"
df_merge.to_csv(path_csv, index_label=False,index=False)

path_excel = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\CLTV_Prediction\\CLTV_Pred_LCW.xlsx"
df_merge.to_excel(path_excel, index=False)


