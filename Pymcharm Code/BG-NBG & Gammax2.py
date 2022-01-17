#### BGNBD  GG İLE CLTV TAHMİNİ VE SONUÇLARIN UZAK SUNUCUYA GÖNDERİLMESİ ###

## ÖNCELİKLE İLGİLİ KÜTÜPHANELERE VE VERİ SETİNE ERİŞİYORUZ.

##################################################
# 1. Veri Hazırlama
##################################################

from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#Verinin Excelden Okunması #tahmin yaptığımız için uç değerleri çözümleme ihtiyacı duyduk

df_ = pd.read_excel("Veri Bilimi/3.Hafta/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df=df_.copy()
df.shape

#Sadece UK dekiler

new_df=df[(df["Country"]== "United Kingdom")]
new_df.describe().T
new_df.head()
#yada database üzerinden credentials için okuyabilirim.
#########################
# Verinin Veri Tabanından Okunması
#########################
# credentials
creds = {'user': 'group_6',
         'passwd': 'miuul',
         'host': '34.79.73.237',
         'port': 3306,
         'db': 'group_6',
        'auth_plugin' : 'mysql_native_password'
}
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}' #to sql kullanması kolay olduğu için
conn = create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)

# top 10 ile veri setimize bakalım :
import pandas as pd
pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

# retail_mysql_df.shape
# retail_mysql_df.head()
# retail_mysql_df.info()
#df=retail_mysql_df.copy()


#Veri Ön İşleme

new_df.dropna(inplace=True)
new_df = new_df[~new_df["Invoice"].str.contains("C", na=False)]
new_df = new_df[new_df["Quantity"] > 0]
new_df = new_df[new_df["Price"] > 0]

replace_with_thresholds(new_df, "Quantity")
replace_with_thresholds(new_df, "Price")

new_df["TotalPrice"] = new_df["Quantity"] * new_df["Price"]

today_date = dt.datetime(2011, 12, 11)

new_df.describe().T

#Lifetime Veri Yapısının hazırlanması

cltv_df = new_df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days, #davranış paterni; ilk ve son işlem arasında geçen süre
                                                         lambda date: (today_date - date.min()).days], #müşterilik yaşı; ilk geldiği ve son geldiği
                                         'Invoice': lambda num: num.nunique(), #kaç işlem yaptığı
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()}) #toplam ciro

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)] #churn eden leri almıyoruz tekrarlı işlem yapanı bul

cltv_df["recency"] = cltv_df["recency"] / 7 #bg-nbd de haftalık cinsten belirtmek isteniyor

cltv_df["T"] = cltv_df["T"] / 7 #müşteri yaşı haftalık sayısı

cltv_df.head()

#Görev1: 6 Aylık CLTV Prediction

bgf.predict(6*4, #4 hafta 1 ay yapar
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_6_month"] = bgf.predict(4*6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_6_month", ascending=False).head(10)

###cltv_df.drop(columns="expected_purc_1_month", axis=1,inplace=True) ihtiyacın olduğunda uçur


# Model objesinin tanımlanması: tahmin edilen kazanç

ggf = GammaGammaFitter(penalizer_coef=0.01) #ceza katsayısı (penalizer_caef) hata payı

# Modelin kurulması:
ggf.fit(cltv_df['frequency'], cltv_df['monetary']) ##uyarlamak(fit etmek)

cltv = ggf.customer_lifetime_value(bgf, #yaşam boyu değeri
                                   cltv_df['frequency'], #neler görmek istersem onları ekle
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık, tahminde bulun zaman kısmı
                                   freq="W",  # T'nin frekans bilgisi.tahminde bulun zaman kısmı
                                   discount_rate=0.01) #indirim oranı burası değişebilir tahminde bulun zaman kısmı

cltv.head()


cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(10)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left") #soldan başlayarak customer ID yi birleştir

cltv_final.sort_values(by="clv", ascending=False).head(10)

#14088 nolu müşterinin M değeri çok yüksek olduğu halde frekansı ve expected_purc_6_month  düşük olduğu
# için kendisinden daha düşük ciro yaratan müşteriden daha altta kalmıştır. Müşteri yaşınada bak. eksi müşteri mi yeni müşteri mi? (T)
#T aynı veriye sabitlediğinde sıklık değeri ne kadara gelir oran orantı ile bul ve o zaman F kısmı ile yorumlayabilirsin
#       Customer ID  recency       T  frequency  monetary  expected_purc_6_month        clv
# 2486   18102.0000  52.2857 52.5714         60 3584.8878                22.8061 85913.5790
# 589    14096.0000  13.8571 14.5714         17 3159.0771                16.7786 56038.8524
# 2184   17450.0000  51.2857 52.5714         46 2629.5299                17.5988 48671.8386
# 2213   17511.0000  52.8571 53.4286         31 2921.9519                11.9822 36891.0488
# 1804   16684.0000  50.4286 51.2857         28 2120.0470                11.2509 25147.3839
# 406    13694.0000  52.7143 53.4286         50 1267.3626                18.8601 25133.3783
# 587    14088.0000  44.5714 46.1429         13 3859.6015                 6.1143 25055.0226
# 1173   15311.0000  53.2857 53.4286         91  667.5968                33.7665 23666.7875
# 133    13089.0000  52.2857 52.8571         97  605.1866                36.2063 23001.5968
# 1485   16000.0000   0.0000  0.4286          3 2055.7867                 9.3767 21374.5234

scaler = MinMaxScaler(feature_range=(0, 1)) ## CLV çok yüksek olduğunda 0,1 arasındaki değerlere getirirse yorumlama kısmı daha kolay olur
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

#Sıralama en kıymetli müşteriler
cltv_final.sort_values(by="scaled_clv", ascending=False).head()


##############################################################
# Görev2: CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################

#1 Ay içinde en çok satın alma beklediğimiz 10 müşteri

bgf.predict(4, #4 hafta 1 ay yapar
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_1_month", ascending=False).head(10)

#12 Ay içinde en çok satın alma beklediğimiz 10 müşteri

bgf.predict(4 * 12,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_12_month"] = bgf.predict(4*12,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])



## 1 aylık ve 12 aylık karşılaştırma
cltv_df.sort_values("expected_purc_12_month", ascending=False).head(15)
cltv_df.sort_values("expected_purc_1_month", ascending=False).head(15)

#bende değişiklik olmadı ancak olsaydı şunlardan kaynaklı olabilir; işlem sayısı artabilir (dönem arttığı için) ancak daha düşük M işlemleri yapılmıştır.
#dolayısı ile clv etkileri düşebilir.

##############################################################
# Görev3: CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################


cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()


cltv_final.sort_values(by="scaled_clv", ascending=False).head(10)


cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})

# a ve b segmente odaklan. çok iyi ciro yaratan müşterileri tutundurmak için sadakat programları uygulanabilir
# özel hediye çeki, ürün hediyeleri olabilir, ilk ürünler en önce bu kullanıcılara önerilir.
#b segmentinde olup a segmentinde olacak yukarı çıkarılabilecek upgrade edebilecek örneğin clct değerleri en yüksek olanlara özel indirim, bir alana bir bedeva kampanyaları yapılabilinir.

##############################################################
# Görev4: Sonuçların Veri Tabanına Gönderilmesi
##############################################################

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)


cltv_final.head()

cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)

cltv_final.to_sql(name='serapgur', con=conn, if_exists='replace', index=False)



pd.read_sql_query("select * from serapgur limit 10", conn)
