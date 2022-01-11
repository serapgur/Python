###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.


# Veri Seti: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

#buna yönelik olarak müşterilerin davranışlarını tanımlayacağız ve
# bu davranışlarda öbeklenmelere göre gruplar oluşturucaz
#yani ort. davranışlar sergileyenleri aynı gruplara alacağız ve
# bu gruplara özel satış ve pazarlama teknikleri geliştirmeye çalışıcaz
#online retail II isimli veri seti ingiltere merkezli online bir satış mağazasının
#01/12/2009-09/12/2011 tarihleri arasındaki satışları içeriyor

# bu şirket hediyelik eşya satıyor. promosyon ürünler gibi düşünebilir

# müşterilerinin çoğu da toptancı

# Değişkenler
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

#Görev 1:
###############################################################
# 1. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_= pd.read_excel("Haftalık Dersler/3.Hafta/Ders Öncesi Notlar/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df= df_.copy()
df.head()

#           InvoiceDate   Price  Customer ID         Country
# 0 2010-12-01 08:26:00 2.55000  17850.00000  United Kingdom
# 1 2010-12-01 08:26:00 3.39000  17850.00000  United Kingdom
# 2 2010-12-01 08:26:00 2.75000  17850.00000  United Kingdom
# 3 2010-12-01 08:26:00 3.39000  17850.00000  United Kingdom
# 4 2010-12-01 08:26:00 3.39000  17850.00000  United Kingdom

df.shape  #kaç değişken kaç gözlem var
df.describe().T  #istatistik değişkenler

#                    count        mean        std          min         25%  \
# Quantity    541910.00000     9.55223  218.08096 -80995.00000     1.00000
# Price       541910.00000     4.61114   96.75977 -11062.06000     1.25000
# Customer ID 406830.00000 15287.68416 1713.60307  12346.00000 13953.00000

#eksik gözlemleri veri setinden çıkartma

df.dropna(inplace=True)
df.isnull().sum()

#eşsiz ürün sayısı
df["Description"].nunique()

## hangi üründen kaçar tane var?
df["Description"].value_counts().head()

## en çok sipariş edilen ürün hangisi? en çok sipariş edilen 5 ürünü çoktan aza doğru sırala
df.groupby("Description").agg({"Quantity": "sum"}).head()
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity" , ascending=False).head()

#Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.
df= df[~df["Invoice"].str.contains("C", na=False)]
df.head()

#Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz.

df = df[(df['Quantity']> 0)]
df = df[(df['Price']> 0)]
df.head()
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()


#Görev2: RFM Metriklerinin Hesaplanması

# Recency (yenilik): Müşterinin son satın almasından bugüne kadar geçen süre
# Frequency (Sıklık): Toplam satın alma sayısı.
# Monetary (Parasal Değer): Müşterinin yaptığı toplam harcama.

df["InvoiceDate"].max()
today_date = dt.datetime(2010, 12, 11)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()
rfm.columns = ['recency', 'frequency', 'monetary']

rfm.head()
rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]

#Görev3: RFM Skorlarının Hesaplanması

#Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
#Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

#Oluşan 2 farklı değişkenin değerini tek bir değişken olarak ifade ediniz ve RFM_SCORE olarak kaydediniz.

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm.head()
rfm[rfm["RFM_SCORE"] == "55"].head()
rfm[rfm["RFM_SCORE"] == "22"].head()

#Görev4:RFM skorlarının segment olarak tanımlanması
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
rfm.head()

#Görev5: Aksiyon  Zamanı !
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
rfm.describe().T

new_df = pd.DataFrame()
new_df["loyal_customers"] = rfm[rfm["segment"] == "loyal_customers"].index
new_df.head()
new_df.describe().T

new_df.to_csv("loyal_customers.csv")

## champions  totalde getirdiği ciro 6857,963 birim para. ziyaret sıklığı 12.41
rfm[rfm["segment"] == "champions"].head()
## Kişinin geçmiş alışverişlerine bakarak yeni gelen ürünlerden bir emailing yapılabilinir.
## 500 birim para alışveriş yaptıklarında min %15 indirim kodu verilebilinir. her kategoride geçerli


##cant_loose ve loyal_customers champions lardan sonra en iyi birim parayı getirmektedir.