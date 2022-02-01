##################################
# AB TEST PROJECT
##################################

### İş Problemi:
# Müşterilerimizden biri olan
# bombabomba.com , bu yeni özelliği test
# etmeye karar verdi ve averagebidding’in, maximumbidding’den daha
# fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak
# istiyor.

### Değişkenler:
# Impression - Reklam Görüntülenme Sayısı
# Click - Tıklama
# Purchase - Satın Alım
# Earning - Kazanç

### Kütüphaneler, Ayarlar, Veri Seti:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

control_df = pd.read_excel("Veri Bilimi/5.Hafta/Ders Öncesi Notlar/ab_testing.xlsx", sheet_name="Control Group")
test_df = pd.read_excel("Veri Bilimi/5.Hafta/Ders Öncesi Notlar/ab_testing.xlsx", sheet_name="Test Group")

#Control_df halihazırda kullanılan maximum bidding yöntemi, test_df ise yeni ürün olan average bidding'i temsil ediyor. Bu ikisini birleştirelim.
control_df["type"] = "control" #iki df birleştiğinde hangisi control hangisi test olduğu belli olsun diye isim verdik.
test_df["type"] = "test"

df = pd.concat([control_df,test_df], axis=0, join="inner")
df.head()
df.groupby("type").agg({"Purchase":"mean"})

#          Purchase
# type
# control 550.89406
# test    582.10610
# Sonuçlar incelendiğinde test setinin satın alma sayısındaki ortalaması daha büyük duruyor.
# AB testimizi yapmadık ancak ilk izlenimimiz yeni yöntem olan averagebidding'in olumlu olduğu yönünde.


### Görev 1: A/B Hipotez Testini Tanımlayınız. (Averagebidding ile maximumbidding'in getirdiği dönüşümlerin fazlalığı arasında istatistiki olarak anlamlı bir sonuç var mıdır?

## Hipotez Testi:

# HO: μ1 = μ2 (Averagebidding ile maximumbidding'in getirdiği dönüşümler arasında anlamlı bir fark yoktur.)
# H1: μ1 != μ2 (Averagebidding ile maximumbidding'in getirdiği dönüşümler arasında anlamlı bir fark vardır.)

# Adımlar:
# Öncelikle varsayım kontrolleri olan normallik varsayımı ve varyans homojenliği test edilecek.
# p-value değeri 0.05'ten büyük gelirse H0 reddedilmeyecek ve parametrik test uygulanacak.
# Şayet burada pvalue değeri 0.05'ten küçük gelirse HO reddedilecek ve non-parametrik test olan mannwhitneyu testi uygulanacak.

## Varsayım Kontrollerinin Yapılması:

# Normallik Varsayımı:

test_stat, pvalue = shapiro(df.loc[df["type"] == "control", "Purchase"])
print("Test Stat: %.4f, p-value: %.4f" % (test_stat,pvalue)) ##Test Stat: 0.9773, p-value: 0.5891 > 0.05

test_stat, pvalue = shapiro(df.loc[df["type"] == "test", "Purchase"])
print("Test Stat: %.4f, p-value: %.4f" % (test_stat,pvalue)) ##Test Stat: 0.9589, p-value: 0.1541 > 0.05

# Her iki durumda da p-value değerleri 0.05'ten büyük geldiği için HO reddedilemez diyoruz, yani normal dağılım varsayım sağlanmaktadır.

# Varyans Homojenliği Varsayımı:

test_stat, pvalue = levene(df.loc[df["type"] == "control", "Purchase"],
                           df.loc[df["type"] == "test", "Purchase"])
print("Test Stat: %.4f, p-value: %.4f" % (test_stat,pvalue)) ##Test Stat: 2.6393, p-value: 0.1083 > 0.05

# Varyans Homojenliği varsayımında da p-value değeri 0.05'ten büyük geldi. HO reddedilemez, varyanslar homojendir.

## Varsayım kontrolümüzde HO'ı reddedilemez bulduğumuz için Parametrik test olan bağımsız iki örneklem T testini yapacağız.

test_stat, pvalue = ttest_ind(df.loc[df["type"] == "control", "Purchase"],
                           df.loc[df["type"] == "test", "Purchase"],
                              equal_var=True)
print("Test Stat: %.4f, p-value: %.4f" % (test_stat,pvalue)) ##Test Stat: -0.9416, p-value: 0.3493 >0.05


## Görev 2: Çıkan test sonuçlarının istatistiksel olarak anlamlı olup olmadığını yorumlayınız.

# Yorum: Verisetindeki odağımız Purchase değerleri olduğu için Bağımsız İki Örneklem T Testi uyguladık.
# Fakat burada diğer değişkenlerin değerlerini göz ardı etmiş olduk.
# Örneğin görüntülenme sayısı olan Impression, tıklama sayısı olan Click değişkenlerinin control ve test kırılımlarındaki durumlarını inceleyelim.

df.groupby("type").agg({"Impression":"mean"})

#           Impression
# type
# control 101711.44907
# test    120512.41176
# Görüldüğü üzere görüntülenme sayılarının ortalamalarında ciddi bir fark var. test verileri yani
# averagebidding yöntemi burada öne çıkıyor.

df.groupby("type").agg({"Click":"mean"})

#              Click
# type
# control 5100.65737
# test    3967.54976

## Burada ise control seti yani maximumbidding yönteminin tıklanma ortalaması test setinin epeyce üstünde yer alıyor.
# Purchase ortalamaları ise birbirine çok yakındı.

# Daha makul sonuç çıkarmak adına, satın alma başına kazanç (Earning/Purchase) ve görüntülenme başına tıklanma oranı (Click/Impression) oranlayıp tekrardan İki Örneklem Oran testi yapalım.

##### İki örneklem Oran Testi:

##Öncelikle hipotezi Satın Alma ve Kazanç üstüne kuralım.

# H0: P1=P2 (maximumbidding ve averagebidding yöntemlerinin satın alma başına kazanç oranları arasında istatistiki olarak anlamlı bir fark yoktur.)
# H1: P1 != P2 (maximumbidding ve averagebidding yöntemlerinin satın alma başına kazanç oranları arasında istatistiki olarak anlamlı bir fark vardır.)

def percent(w1,w2):
    return w1/w2

df["average_purchase"] = df.apply(lambda x: percent(x["Earning"],x["Purchase"]),axis=1)

df.loc[df["type"] == "control","average_purchase"].mean() #3.688075003567011
df.loc[df["type"] == "test","average_purchase"].mean() #4.652948745118226

control_purchase_count = df.loc[df["type"] == "control", "Purchase"].sum()
test_purchase_count = df.loc[df["type"] == "test", "Purchase"].sum()

control_earning_sum = df.loc[df["type"] == "control", "Earning"].sum()
test_earning_sum = df.loc[df["type"] == "test", "Earning"].sum()


test_stat, pvalue = proportions_ztest(count=[control_purchase_count,test_purchase_count],
                                      nobs=[control_earning_sum,test_earning_sum])

print("Test Stat: %.4f, p-value: %.4f" % (test_stat,pvalue)) #Test Stat: 27.2908, p-value: 0.0000 #Satın alma-kazanç ilişkisi incelendiğinde control ve test arasında fark vardır.

##Şimdide hipotezi görüntülenme ve tıklanma üstüne kuralım.

# H0: P1=P2 (maximumbidding ve averagebidding yöntemlerinin görüntülenme başına tıklanma oranları arasında istatistiki olarak anlamlı bir fark yoktur.)
# H1: P1 != P2 (maximumbidding ve averagebidding yöntemlerinin görüntülenme başına tıklanma oranları arasında istatistiki olarak anlamlı bir fark vardır.)

def percent(w1,w2):
    return w1/w2

df["average_click"] = df.apply(lambda x: percent(x["Click"],x["Impression"]),axis=1)
df.loc[df["type"] == "control","average_click"].mean() #0.05361823086521901
df.loc[df["type"] == "test","average_click"].mean() #0.034175991543627396

control_click_count = df.loc[df["type"] == "control", "Click"].sum()
test_click_count = df.loc[df["type"] == "test", "Click"].sum()

control_impression_sum = df.loc[df["type"] == "control", "Impression"].sum()
test_impression_sum = df.loc[df["type"] == "test", "Impression"].sum()

test_stat, pvalue = proportions_ztest(count=[control_click_count,test_click_count],
                                      nobs=[control_impression_sum,test_impression_sum])

print("Test Stat: %.4f, p-value: %.4f" % (test_stat,pvalue)) #Test Stat: 129.3305, p-value: 0.0000 #Görüntülenme-tıklanma ilişkisi incelendiğinde control ve test arasında fark vardır.

### Görev 3: Hangi testleri kullandınız? Sebeplerini belirtiniz.

# Varsayım kontrollerinde: Normallik varsayımına bakılırken Shapiro Wilk testi, Varyans Homojenliği kısmında Levene testi kullanıldı.
# İlk yaptığımız AB testinde normallik varsayımı ve varyans homojenliği sağlandığı için Tek Örneklem T testi kullanıldı. Eğer varsayım kontrollerinde HO reddedilmiş olsaydı Mannwhitneyu kullanılırdı.

# İkinci AB testinde ise İki Örneklem Oran Testi yani Z testi kullanıldı. Bunun sebebi yukarıdaki aşamalarda da belirtildiği üzere Purchase değerlerinin control ve test gruplarında birbirine
# ortalama ve toplam olarak benzer davranışlar gösteriyor olması fakat, Impression, Click ve Earning değişkenlerinin göz ardı edilmesinin mantıklı olmamasıdır.

### Görev 4: Görev 2’de verdiğiniz cevaba göre, müşteriye tavsiyeniz nedir?

# Müşterinin her iki type'da Satın alma ve harcama alışkanlıkları incelendiğinde A/B testi sonucu fark olduğu gözlemlemiştir.
# Test setinde yani yeni yöntem olaran average bidding'de müşteriler satın alma başına daha çok harcama yapmaktadır. Kazanç anlamında da test grubunun total getirisi daha çok kazanç sağlandığını göstermektedir.

# Fakat görüntülenme başı tıklama oranlarına bakıldığında ise control grubunda her görüntülenme başına tıklanma oranı, test grubuna göre daha fazladır.
# Bu da müşterimizin yani bombabomba.com isimli sitenin maximum bidding yöntemi ile sitesine daha çok trafik çektiğini göstermektedir.

# Burada önemli olan müşterinin talebidir, şayet müşterimiz kazancı ön plana koyuyorsa average bidding'i, trafiği ön plana koyuyorsa maximum bidding'i tercih edebilir.















































