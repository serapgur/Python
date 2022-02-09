################################
# Customer Segmentation with K-Means
################################

# İş Problemi
# Kural tabanlı müşteri segmentasyonu
# yöntemi RFM ile makine öğrenmesi yöntemi
# olan K-Means'in müşteri segmentasyonu için
# karşılaştırılması beklenmektedir

##  Veri seti hikayesi

# Online Retail II isimli veri seti İngiltere merkezli online bir satış
# mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını
# içermektedir.
# Bu şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır.
# Şirketin müşterilerinin büyük çoğunluğu kurumsal müşterilerdir.

## Değişkenler

# InvoiceNO – Fatura Numarası
# Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.
# StockCode – Ürün Kodu
# Her bir ürün için eşsiz numara
# Description – Ürün İsmi
# Quantity – Ürün Adedi
# Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate – Fatura tarihi
# UnitPrice – Fatura fiyatı (Sterlin)
# CustomerID – Eşsiz müşteri numarası
# Country – Ülke ismi


# RFM metriklerine göre (skorlar değil) K-Means'i kullanarak müşteri
# segmentasyonu yapınız.
# Dilerseniz RFM metriklerinden başka metrikler de üretebilir ve bunları da
# kümeleme için kullanabilirsiniz

# Ortak davranış sergileyen müşterileri aynı gruplara alacağız ve bu gruplara özel satış ve pazarlama teknikleri geliştirmeye çalışacağız.
#kütüphanelerin kurulumu

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import datetime


# Veriyi Anlama

#tüm sütunları ve satırların görüntülenmesi
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None)

#virgulden sonra gösterilecek olan sayı sayısı
pd.set_option('display.float_format', lambda x: '%.0f' % x)

#veri setini okuma

retail = pd.read_excel('Dataset/online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = retail.copy()
df.head()

#en cok siparis edilen urunlerin sıralaması
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()

#toplam kaç fatura sayısı
df["Invoice"].nunique()

#en pahalı ürünler
df.sort_values("Price", ascending = False).head()

#en fazla sipariş sayısına sahip ilk 5 ülke
df["Country"].value_counts().head()

#toplam harcamayı sütun olarak ekledik
df['TotalPrice'] = df['Price']*df['Quantity']

#hangi ülkeden ne kadar gelir elde edildi
df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()

#Veri Ön İşleme

#en eski alışveriş tarihi
df["InvoiceDate"].min()

# Timestamp('2010-12-01 08:26:00')

#en yeni alışveriş tarihi
df["InvoiceDate"].max()

#Timestamp('2011-12-09 12:50:00')

#değerlendirmenin daha kolay yapılabilmesi için bugünün tarihi olarak 1 Ocak 2012 tarihi belirlendi.

today = pd.datetime(2011, 12, 11)
today

#sipariş tarihinin veri tipinin değiştirilmesi
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

#0'dan büyük değerleri alınması, bu işlem değerlendirmeyi daha kolaylaştıracak
df = df[df['Quantity'] > 0]
df = df[df['TotalPrice'] > 0]

#eksik verilere sahip gözlem birimlerinin df üzerinden kaldırılması
df.dropna(inplace = True)

#veri setinde eksik veri yok
df.isnull().sum(axis=0)

#boyut bilgisi
df.shape
#out=(397885, 9)

df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T
#belirtilen yüzdelere karşılık gelen gözlem birimlerinin açıklayıcı istatistik değerleri
#değerlendirmeyi kolaylaştırması amacıyla df tablosunun transpozu alındı.

#            count  mean  std   min    1%    5%   10%   25%   50%   75%   90%  \
# Quantity    397885    13  179     1     1     1     1     2     6    12    24
# Price       397885     3   22     0     0     0     1     1     2     4     6
# Customer ID 397885 15294 1713 12346 12415 12627 12883 13969 15159 16795 17725
# TotalPrice  397885    22  309     0     1     1     2     5    12    20    35

# İngiltere merkezli old. için yalnızca Birleşik Krallık verilerini tutma

df_uk = df.query("Country=='United Kingdom'").reset_index(drop=True)

df_uk.head()

# RFM; Recency, Frequency, Monetary ifadelerinin ilk harflarinden oluşur.
# Müşterilerin satın alma alışkanlıkları üzerinden pazarlama ve satış stratejileri belirlemeye yardımcı olan bir tekniktir.

# Recency (yenilik): Müşterinin son satın almasından bugüne kadar geçen süre
#
# Frequency (sıklık): Toplam satın alma sayısı.
#
# Monetary (parasal değer): Müşterinin yaptığı toplam harcama.

df.head()

df.info()
#dataframe'in indeks tipleri, sütun türleri, boş olmayan değerler ve bellek kullanım bilgileri

#Recency ve Monetary değerlerinin bulunması
df_x = df.groupby('Customer ID').agg({'TotalPrice': lambda x: x.sum(), #monetary value
                                        'InvoiceDate': lambda x: (today - x.max()).days}) #recency value
#x.max()).days; müşterilerin son alışveriş tarihi

#kişi başına düşen frequency değerini bulunması
df_y = df.groupby(['Customer ID','Invoice']).agg({'TotalPrice': lambda x: x.sum()})
df_z = df_y.groupby('Customer ID').agg({'TotalPrice': lambda x: len(x)})

#RFM tablosunun oluşturulması
rfm_table= pd.merge(df_x,df_z, on='Customer ID')

#Sütun isimlerini belirlenmesi
rfm_table.rename(columns= {'InvoiceDate': 'Recency',
                          'TotalPrice_y': 'Frequency',
                          'TotalPrice_x': 'Monetary'}, inplace= True)

#RFM Tablosu
rfm_table.head()

### K-Means Segmentasyonu
# Gözetimsiz öğrenme algoritmalarından biridir. Yapılan işlem, verileri kendine özgü özelliklerine göre ayırmak ve karakteristik özelliklerini ortaya çıkarmaktır.
# Amaç, noktalar arasındaki toplam mesafeyi en aza indirmek ve kümeler arasındaki mesafeyi en üst düzeye çıkarmaktır. Bu süreç kümelemedir.

# Kümeleme nerede kullanılır?
#
#  - Ürünlerin satın alan müşteri gruplarına göre kümelenmesi
#  - Belgelerin kümelenmesi, kullanılan benzer kelimelere göre web aramaları
#  - Müşteri segmentasyonu
#  - Biyoinformatik alanında benzer genlerin gruplanması

# Recency, Frequency and Monetary Değerlerinin Görselleştirilmesi
# Recency için tanımlayıcı istatistikler

rfm_table.Recency.describe()

#Recency dağılım grafiği

x = rfm_table['Recency']

ax = sns.distplot(x)

#Frequency için tanımlayıcı istatistikler
rfm_table.Frequency.describe()

#Frequency dağılım grafiği, 1000'den az Frequency değerine sahip gözlemlerin alınması

x = rfm_table.query('Frequency < 1000')['Frequency']

ax = sns.distplot(x)

#Monetary için tanımlayıcı istatistikler
rfm_table.Monetary.describe()

#Monatary dağılım grafiği, 10000'den az Monetary değerine sahip gözlemlerin alınması
x = rfm_table.query('Monetary < 10000')['Monetary']

ax = sns.distplot(x)

#çeyreklikler kullanılarak dört parçaya bölünme işlemi
quantiles = rfm_table.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()

#Dönüştürme işlemi
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler((0,1))
x_scaled = min_max_scaler.fit_transform(rfm_table)
data_scaled = pd.DataFrame(x_scaled)
#burada değerlerimizi normalleştirdik

df[0:10]

plt.figure(figsize=(8,6))
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',n_init=10, max_iter = 300)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Burada önemli olan doğru küme sayısını belirlemektir. Bunun için Elbow Method'unu uygularız.

# Burada küme sayısı arttıkça, dirsek dediğimiz kısmı kullanıyoruz ve belli bir noktadan sonra değerin çok fazla düşmediğini görüyoruz.

# Bu nedenle 3 veya 4 kullanmak bizim için daha değerli olabilir.

kmeans = KMeans(n_clusters = 4, init='k-means++', n_init =10,max_iter = 300)
kmeans.fit(data_scaled)
cluster = kmeans.predict(data_scaled)
#init = 'k-means ++' bu daha hızlı çalışmasını sağlar

d_frame = pd.DataFrame(rfm_table)
d_frame['cluster_no'] = cluster
d_frame['cluster_no'].value_counts() # küme başına kişi sayısı (Custer ID numarası)

rfm_table.head()

#kümelerin ortalama değerleri
d_frame.groupby('cluster_no').mean()

# Sonuç

# Bu çalışmada RFM metrikleri üzerinden 3 değişkene dayalı olarak Customer ID değerleri tekilleştirilmiştir.
# Bu veri seti, k-means yöntemi uygulanarak sayısal değişkenler temelinde 4 kümeye indirilmiştir.