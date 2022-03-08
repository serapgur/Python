############################
# Setting Churn w/0-1
############################

import pandas as pd
from datetime import datetime
from datetime import timedelta

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df = pd.read_csv("Proje/Studies/Studies w_Datav2/TENURE/Tenure_LCW.csv")
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

##################
# Churn Değerlerinin Bulunması
##################
df["OrderDate"] = df["OrderDate"].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))


df["OrderDate"].max() - df["OrderDate"].min()
# Veri setimiz içerisinde 730 günlük veri var.

df.describe([0.1,0.25,0.5,0.75,0.9]).T

# Describe incelendiğinde Recency değişkeninin min değeri 1, max değeri 726, ortalaması 65.66,%25 değeri 10, %50 değeri 27, %75 değeri ise 68 çıkmaktadır.
df["recency_score"].value_counts()
# 5    41686
# 4    24751
# 3    17142
# 2     9762
# 1     5707


# Recency score'u azaldıkça yani son alışveriş tarihi ile analiz yapılan tarihler arasındaki süre uzadıkça kişi sayısının azaldığını görüyoruz.
df[df["recency_score"] == 5]["recency"].min() # 1 ile 19 gün arasına 5 denilmiş
df[df["recency_score"] == 4]["recency"].max() # 20 ile 44 gün arasına 4 denilmiş
df[df["recency_score"] == 3]["recency"].max() # 45 ile 112 gün arasına 3 denilmiş
df[df["recency_score"] == 2]["recency"].max() # 113 ile 268 gün arasına 2 denilmiş
df[df["recency_score"] == 1]["recency"].max() # 267 ile 726 gün arasına 1 denilmiş

# Recency değerlerine göre recency score'u 1 olan yani 267 gün ve daha öncesinde alışveriş yapan müşterilere kesinlikle churn diyebiliriz.
# Diğer metrikleri de değerlendirerek score'u 2 ve 3 olanları da nasıl katacağımıza bakalım.

# Describe incelendiğinde Frequency değişkeninin min değeri 1, max değeri 126, ortalaması 20.23,%25 değeri 6, %50 değeri 13, %75 değeri ise 27 çıkmaktadır.

df["frequency_score"].value_counts()
# 5    65949
# 4    18495
# 3     8456
# 2     3655
# 1     2493
df[df["frequency_score"] == 1]["frequency"].max() # Sadece 1 kere alışveriş yapanlar
df[df["frequency_score"] == 2]["frequency"].max() # Sadece 2 kere alışveriş yapanlar
df[df["frequency_score"] == 3]["frequency"].max() # Sadece 3 ve 4 kere alışveriş yapanlar
df[df["frequency_score"] == 4]["frequency"].max() # 4-8 arası alışveriş yapanlar
df[df["frequency_score"] == 5]["frequency"].max() # 8-126 arası alışveriş yapanlar

# Frequency değişkenine bakıldığında 1 ve 2 kere alışveriş yapan kişilerin kişi sayısının sıralamada sonlarda olduğunu görüyoruz.


# Describe incelendiğinde Monetary değişkeninin min değeri 7.95, max değeri 16859.76, ortalaması 3005.53,%25 değeri 857.75, %50 değeri 1890, %75 değeri ise 4154.81 çıkmaktadır
df["monetary_score"].value_counts()
# 5    68226
# 4    17705
# 3     7890
# 2     3540
# 1     1687

df[df["monetary_score"] == 1]["monetary"].max() # 99.99 TL'ye kadar para harcamış olanlar
df[df["monetary_score"] == 2]["monetary"].max() # 100.13 - 220.92 TL arası para harcamış olanlar
df[df["monetary_score"] == 3]["monetary"].min() # 221.85 - 470.52 TL arası para harcamış olanlar
df[df["monetary_score"] == 4]["monetary"].min() # 470.89 - 1093.13 TL arası para harcamış olanlar
df[df["monetary_score"] == 5]["monetary"].min() # 1094.33 - 16859.76 TL arası para harcamış olanlar




# Describe incelendiğinde Tenure değişkeninin min değeri 1, max değeri 730, ortalaması 555.38,%25 değeri 442, %50 değeri 635, %75 değeri ise 706 gün çıkmaktadır.

# Describe incelendiğinde CLTV değişkeninin min değeri 0.0176, max değeri 79074.36, ortalaması 5112.54,%25 değeri 204.67, %50 değeri 993.92, %75 değeri ise 4802.14 çıkmaktadır.


"""
Churn kararı: 
Recency Score'u 1 olan yani en son 267 gün ve öncesinde olan müşterilerin tamamı churn kabul edilebilir. 
Frequency score'u 2 ve altında olan, monetary score'u 2 ve altında olan, CLTV değeri ortalamanın altında kalan ve tenure değeri ortalamanın altında kalan recency 2 scorelarını churn kabul edebiliriz. 
Recency 3 score'u 45-112 gün arasını ifade etmekteydi. Tamamen 3'ü kapsayan kitle içinde churn değerlendirmesi yapmak doğru olmayabilir. 
Bu sebeple recency değeri 90-112 arası olan kitlede, frequency score'u 1 olan, monetary score'u 1 olan, tenure değeri %25'lik değerin altında kalan, CLTV değeri ise %25'in altında kalan kişileri churn olarak alabiliriz. 
"""

# Churn olanları 1, olmayanları 0 kabul edelim.
df.loc[(df["recency_score"]==1),"Churn"] = 1
df.loc[(df["recency_score"]==2) & (df["frequency_score"]<=2) & (df["monetary_score"]<=2)  & (df["CLTV"] <= 5112) & (df["T"] <= 555.38), "Churn"] = 1
df.loc[(df["recency"]<=112) & (df["recency"] >= 90) & (df["frequency_score"]==1) & (df["monetary_score"]==1) & (df["CLTV"] <= 204.67) & (df["T"] <= 442), "Churn"] = 1
df["Churn"].fillna(0,inplace=True)
df.head()

df[df["Churn"]==1]["Churn"].count() / df["Churn"].count()*100
# Veri seti içerisinde yapılan alışverişlere bakıldığında tüm alışverişlerin %6.82'si churn olan kişilerin alışverişinden oluşuyor.

musteriler = df.groupby("MusteriRef").agg({"Churn":"max"})
musteriler[musteriler["Churn"]==1].count() / musteriler["Churn"].count()
# Tüm müşterilerin %29,43'ü churn olmuş görünüyor.

# Veri setinin son halini kayıt edelim.

path = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\CHURN\\Churn_LCW.csv"
df.to_csv(path, index_label=False,index=False)

path2 = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\CHURN\\Churn_LCW.xlsx"
df.to_excel(path2, index=False)