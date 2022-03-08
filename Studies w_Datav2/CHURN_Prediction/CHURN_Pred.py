############################
# Churn Prediction with Machine Learning on LCW Customers Data
############################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from pandas.api.types import CategoricalDtype


# Veriyi import edecek fonksiyonu alalım.
def load_LCW():
    data = pd.read_csv("Proje/Studies/Studies w_Datav2/CHURN/Churn_LCW.csv")
    return data

df = load_LCW()

# Check_df ile verinin genel hatlarını gözlemleyelim.

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### ISNULL #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Modelde işimize yaramayacak olan değişkenleri kaldıralım.

df = df.drop(['OrderId',"UrunRef","SiparisYilAyGun","FaturaYilAyGun" ,"Kod","ISO_Kod","DovizRef"], axis=1)


# Değişkenleri kategorik, numerik vb. durumlara göre ayıralım.

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
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

# Numeric ve categoric değişkenlerin doğru yerde olup olmadıklarını kontrol edelim ve gerekirse değişiklik yapalım.

# Numerik değişkenleri kontrol edip yanlış atamaları düzeltelim.
df[num_cols].head()
num_cols = [i for i in num_cols if i not in ["MusteriRef", "OrderDeliveryCityRef", "RFM_SCORE"]]

# Kategorik değişkenleri kontrol edip yanlış atamaları düzeltelim.
df[cat_cols].head()
cat_cols = [i for i in cat_cols if i not in ["TaxRatio"]]

# Numerik değişkenlerden çıkardığımız bazı değişkenleri kategorik değişken olarak ekleyelim.

cat_cols = cat_cols + ["RFM_SCORE"]

# Kategorik değişkenlerden çıkardığımız bazı değişkenleri Numerik değişken olarak ekleyelim.

num_cols = num_cols + ["TaxRatio"]

#####KATEGORİK DEĞİŞKEN ANALİZİ#####

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Kategorik değişkenler arasındaki korelasyonu inceleyelim.

df[cat_cols].corr()


# Hedef değişken Churn'e göre kategorik değişkenlerin kendi içlerindeki dağılımına bakalım.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "Churn", cat_cols)

#####SAYISAL DEĞİŞKEN ANALİZİ#####

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.30, 0.50, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)   #  burada  block=True argümanını üst üste gelmesin diye.

for col in num_cols:
    num_summary(df, col, plot=True)

## Dikkat: Aykırı değer analizini Tenure:T değişkenini oluştururken incelemiştik.
# Tekrardan aykırı değer analizi yapmayacağız çünkü her ne kadar aykırı değerlere sürekli müdehale etsekte her seferinde yeni
# aykırı değerler ortaya çıkacaktır.

# Eksik değerleri gözlemleyelim.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Veri setinde eksik değer olmadığını gözlemliyoruz, bunun içinde herhangi bir işlem yapmamıza gerek kalmadı.


# Sayısal değişkenlerin kendi içlerindeki korelasyonuna göz gezdirelim.

df[num_cols].corr()

# Çok fazla sayısal değişken olduğu için 0.5'in üstündeki korelasyonları inceleyelim.

corr_matrix = df.corr()
kot = corr_matrix[(corr_matrix>=.75) & (corr_matrix !=1)]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Reds")
plt.show()

df.corr().unstack().sort_values().drop_duplicates().sort_values(ascending=False)


# Yeni değişkenler türetelim. (Feauture Eng)

# 1. Şehirlere göre satış toplamlarına bakalım ve bu toplamlara göre şehirlere değer verelim.

cities = df.groupby("SehirAdi").agg({"UnitPrice":"sum"})

cities.sort_values(by="UnitPrice", ascending=False).head(5)

# SehirAdi           UnitPrice

# İSTANBUL         1669565.154
# ANKARA            666745.417
# İZMİR             422190.507
# ANTALYA           211788.987
# MUĞLA              83734.440

# Şehirleri A-E arası satış değerlerine göre kategorilendirelim. (A satışı en yüksek, E en düşük)

cities["NEW_City_Segment_by_Sales"] = pd.qcut(cities["UnitPrice"],5,labels=["E", "D", "C", "B", "A"])
del cities["UnitPrice"]

# Bu segmentleri asıl veri setimize alalım.

df = pd.merge(df,cities,how="left",left_on="SehirAdi",right_on="SehirAdi")
df.head()

# Veri setini burada kayıt edelim.

path_csv = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\CHURN_Prediction\\Lastdf_for_model.csv"
df.to_csv(path_csv, index_label=False,index=False)


# Yeni değişkenlerden sonra kategorik ve numerik değişkenlerimi ayıralım.

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [i for i in num_cols if i not in ["MusteriRef", "OrderDeliveryCityRef", "RFM_SCORE"]]
cat_cols = [i for i in cat_cols if i not in ["TaxRatio"]]
cat_cols = cat_cols + ["RFM_SCORE"]
num_cols = num_cols + ["TaxRatio"]

# Kategorik değişkenlerimizi encode edelim.

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Sayısal değişkenlerimizi standartlaştıralım.

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df.shape # (99048, 88)

df.rename(columns={"Churn_1.0":"Churn"}, inplace=True)
df.head()

######################
# CHURN MODEL
######################

## Base Model

X = df.drop(["Churn","KlasmanGrupTanim","BuyerGrupTanim","OrderDate","SehirAdi"],axis=1)
y= df[["Churn"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 10)




classifiers = [('LR', LogisticRegression(solver='liblinear')),
               ('KNN', KNeighborsClassifier()),
               ("SVC", SVC()),
               ("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('Adaboost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ('XGBoost', XGBClassifier(eval_metric='mlogloss')),
               ('LightGBM', LGBMClassifier()),
               ('CatBoost', CatBoostClassifier(verbose=False))
               ]


for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X_train, y_train, cv=10, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")

# AUC: 1.0 (LR)
# AUC: 0.978 (KNN)
# AUC: 0.9947 (SVC)
# AUC: 0.9997 (CART)
# AUC: 1.0 (RF)
# AUC: 1.0 (Adaboost)
# AUC: 1.0 (GBM)
# AUC: 1.0 (XGBoost)
# AUC: 1.0 (LightGBM)
# AUC: 1.0 (CatBoost)

# Basemodel'de overfit olduk.
# Churn değişkenini kendimiz oluşturduğumuz için model bu ilintiyi algılayabiliyor.
# Buna çözüm için önce ilintide kullandığımız değişkenleri silelim, bir de churn değişkenine gürültü ekleyip deneyelim.

# Bazı değişkenlerin silinerek BASE MODEL Oluşturulması

df =pd.read_csv("Proje/Studies/Studies w_Datav2/CHURN_Prediction/Lastdf_for_model.csv")
df.head()

silinecek_degiskenler = ["recency","frequency","monetary","recency_score","frequency_score", "monetary_score","Total_Transaction",
                         "Total_Unit","Total_Price","Avg_Order_Value","Purchase_Frequency","Profit_Margin","Customer_Value","CLTV","T","RFM_SCORE","segment"]

df.drop(silinecek_degiskenler,axis=1,inplace=True)
df.shape
df.head()

# Düzenlemeden sonra kategorik ve numerik değişkenlerimi ayıralım.

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [i for i in num_cols if i not in ["MusteriRef", "OrderDeliveryCityRef"]]
cat_cols = [i for i in cat_cols if i not in ["TaxRatio"]]
num_cols = num_cols + ["TaxRatio"]

# Kategorik değişkenlerimizi encode edelim.

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Sayısal değişkenlerimizi standartlaştıralım.

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df.shape # (99048, 39)

df.rename(columns={"Churn_1.0":"Churn"}, inplace=True)
df.head()

# Tekrar base modelimizi kurup sonuçlara bakalım.
X = df.drop(["Churn","KlasmanGrupTanim","BuyerGrupTanim","OrderDate","SehirAdi"],axis=1)
y= df[["Churn"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 10)

classifiers = [('LR', LogisticRegression(solver='liblinear')),
               ('KNN', KNeighborsClassifier()),
               ("SVC", SVC()),
               ("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('Adaboost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ('XGBoost', XGBClassifier(eval_metric='mlogloss')),
               ('LightGBM', LGBMClassifier()),
               ('CatBoost', CatBoostClassifier(verbose=False))
               ]


for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X_train, y_train, cv=10, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")

# AUC: 0.5997 (LR)
# AUC: 0.9557 (KNN)
# AUC: 0.5319 (SVC)
# AUC: 0.8183 (CART)
# AUC: 0.8997 (RF)
# AUC: 0.6916 (Adaboost)
# AUC: 0.7596 (GBM)
# AUC: 0.94 (XGBoost)
# AUC: 0.8866 (LightGBM)

# Bazı modellerin başarısı gayet iyi bazılarının ki çok kötü çıktı. Biraz da gürültü ekleyip tamamlayalım.

# Bu kısımdan sonra MLModels.py dosyası  ile devam ediyorum.