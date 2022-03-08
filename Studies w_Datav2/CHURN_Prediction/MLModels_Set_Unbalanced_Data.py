
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

#########
# Veri setinde churn olmayan gözlem sayısı churn olan gözlem sayısına göre çok çok fazla.
# Bu veri setinde dengesizlik yaratıyor.
# Churn olmayan verilerden rastgele veri silelim.

df[df["Churn"] == 0].shape # (92285, 38)
df[df["Churn"] == 1].shape # (6763, 38)

# Churn olmayan müşterilerin sayısını 20000'e düşürelim yani churn olmayan grubu %78.328 oranında azaltalım.

part_frac = df[df["Churn"] == 0].sample(frac=0.7832800563471853)
df = df.drop(part_frac.index)
df.shape # (26763, 38)
df[df["Churn"] == 0].shape # (20000, 38)
df[df["Churn"] == 1].shape # (6763, 38)

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

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

target_summary_with_cat(df,"Churn",cat_cols)


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
        plt.grid(False)
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

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    num_cols = grab_col_names(dataframe)[1]
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu",  annot=True, annot_kws={"size": 9}, fmt= '.2f')
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True, corr_th=0.70)

df.corr().unstack().sort_values().drop_duplicates().sort_values(ascending=False)


# Yeni değişkenler türetelim. (Feauture Eng)

# 1. Şehirlere göre satış toplamlarına bakalım ve bu toplamlara göre şehirlere değer verelim.

cities = df.groupby("SehirAdi").agg({"UnitPrice":"sum"})

cities.sort_values(by="UnitPrice", ascending=False).head(5)

#                   UnitPrice
# SehirAdi
# İSTANBUL         438631.180
# ANKARA           179422.270
# İZMİR            114977.330
# ANTALYA           55928.177
# MUĞLA             23794.430

# Şehirleri A-E arası satış değerlerine göre kategorilendirelim. (A satışı en yüksek, E en düşük)

cities["NEW_City_Segment_by_Sales"] = pd.qcut(cities["UnitPrice"],5,labels=["E", "D", "C", "B", "A"])
del cities["UnitPrice"]

# Bu segmentleri asıl veri setimize alalım.

df = pd.merge(df,cities,how="left",left_on="SehirAdi",right_on="SehirAdi")
df.head()

# Veri setini burada kayıt edelim.

path_csv = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\CHURN_Prediction\\Lastdf_for_model_for_unbalanced.csv"
df.to_csv(path_csv, index_label=False,index=False)



#################################
#Yeni veri seti üzerinden ilerleyelim.

df = pd.read_csv("Proje/Studies/Studies w_Datav2/CHURN_Prediction/Lastdf_for_model_for_unbalanced.csv")


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

X = df.drop(["Churn","KlasmanGrupTanim","BuyerGrupTanim","OrderDate","SehirAdi","MusteriRef"],axis=1)
y= df[["Churn"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 10)




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
    cv_results = cross_validate(classifier, X_train, y_train, cv=5, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")

# AUC: 1.0 (LR)
# AUC: 0.9956 (KNN)
# AUC: 0.9986 (SVC)
# AUC: 0.9994 (CART)
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

df =pd.read_csv("Proje/Studies/Studies w_Datav2/CHURN_Prediction/Lastdf_for_model_for_unbalanced.csv")
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
X = df.drop(["Churn","KlasmanGrupTanim","BuyerGrupTanim","OrderDate","SehirAdi","MusteriRef"],axis=1)
y= df[["Churn"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 10)

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
    cv_results = cross_validate(classifier, X_train, y_train, cv=5, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")

# AUC: 0.5945 (LR)
# AUC: 0.6923 (KNN)
# AUC: 0.6125 (SVC)
# AUC: 0.6626 (CART)
# AUC: 0.756 (RF)
# AUC: 0.6738 (Adaboost)
# AUC: 0.7192 (GBM)
# AUC: 0.8265 (XGBoost)
# AUC: 0.7808 (LightGBM)
# AUC: 0.7758 (CatBoost

# Bazı modellerin başarısı gayet iyi bazılarının ki çok kötü çıktı. Biraz da gürültü ekleyip tamamlayalım.
df = pd.read_csv("Proje/Studies/Studies w_Datav2/CHURN_Prediction/Lastdf_for_model_for_unbalanced.csv")
df.head()


# Gürültüyü müşteri bazında ayrıştırarak ekleyelim.

musteriler = df.groupby("MusteriRef").agg({"Churn":"max"})
musteriler.head()

musteriler[musteriler["Churn"]==1].count() / musteriler["Churn"].count()
# Gürültü eklemeden müşterilerin Churn olma oranı %33.1

# En son alışverişini 60 gün ve daha fazla süre önce alışveriş yapan müşterilere belirli periyotlarda gürültü ekleyelim.

# Gürültü ekleyecek fonksiyonumuzu yazalım.
def add_noise(day1, frac, day2=0):
    print("Gürültü eklenmeden önce veri setinde churn olan gözlemlerin tüm gözlemlere oranı: "+str(df[df["Churn"] == 1]["Churn"].count() / df["Churn"].count() * 100))

    if day2==0:
        musteriler = df[(df["recency"] >= day1)].groupby("MusteriRef").agg({"Churn": "max"})
        musteriler[musteriler["Churn"] == 1].count() / musteriler["Churn"].count()
        musteriler = musteriler[musteriler["Churn"] == 0]
        part_frac = musteriler.sample(frac=frac)
        musteriler_returnto_churn = musteriler.drop(part_frac.index)
        index_changed = musteriler_returnto_churn.index
        df.loc[df["MusteriRef"].isin(index_changed), "Churn"] = 1
    else:
        musteriler = df[(df["recency"] >= day1) & (df["recency"] < day2)].groupby("MusteriRef").agg({"Churn": "max"})
        musteriler[musteriler["Churn"] == 1].count() / musteriler["Churn"].count()
        musteriler = musteriler[musteriler["Churn"] == 0]
        part_frac = musteriler.sample(frac=frac)
        musteriler_returnto_churn = musteriler.drop(part_frac.index)
        index_changed = musteriler_returnto_churn.index
        df.loc[df["MusteriRef"].isin(index_changed), "Churn"] = 1
    print("Gürültü eklendikten sonra veri setinde churn olan gözlemlerin tüm gözlemlere oranı: "+str(df[df["Churn"] == 1]["Churn"].count() / df["Churn"].count() * 100))

#90 gün ve daha öncesi-%20 gürültü
add_noise(90,0.8)
# Gürültü eklenmeden önce veri setinde churn olan gözlemlerin tüm gözlemlere oranı: 25.269962261330942
# Gürültü eklendikten sonra veri setinde churn olan gözlemlerin tüm gözlemlere oranı: 27.38482232933528

#75-90 gün arası-%25 gürültü
add_noise(75,0.75,90)
# Gürültü eklenmeden önce veri setinde churn olan gözlemlerin tüm gözlemlere oranı: 27.38482232933528
# Gürültü eklendikten sonra veri setinde churn olan gözlemlerin tüm gözlemlere oranı: 28.023764152000897

#60-75 gün arası-%25 gürültü
add_noise(60,0.75,75)
# Gürültü eklenmeden önce veri setinde churn olan gözlemlerin tüm gözlemlere oranı: 28.023764152000897
# Gürültü eklendikten sonra veri setinde churn olan gözlemlerin tüm gözlemlere oranı: 28.94668011807346

# Churn değişkenini belirlerken kullandığımız değişkenleri silelim. (Tenure değişkeni tutuldu.)

silinecek_degiskenler = ["recency","frequency","monetary","recency_score","frequency_score", "monetary_score","Total_Transaction",
                         "Total_Unit","Total_Price","Avg_Order_Value","Purchase_Frequency","Profit_Margin","Customer_Value","CLTV","RFM_SCORE","segment"]

df.drop(silinecek_degiskenler,axis=1,inplace=True)
df.shape
df.head()


# Gürültümüzü ekledik, değişkenlerimizi sildik. Base modeli tekrar kuralım.

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [i for i in num_cols if i not in ["MusteriRef", "OrderDeliveryCityRef"]]


# Kategorik değişkenlerimizi encode edelim.

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Sayısal değişkenlerimizi standartlaştıralım.

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df.shape

df.rename(columns={"Churn_1.0":"Churn"}, inplace=True)
df.head()

# Tekrar base modelimizi kurup sonuçlara bakalım.
X = df.drop(["Churn","KlasmanGrupTanim","BuyerGrupTanim","OrderDate","SehirAdi","MusteriRef"],axis=1)
y= df[["Churn"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 10)

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
    cv_results = cross_validate(classifier, X_train, y_train, cv=5, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")

# AUC: 0.6479 (LR)
# AUC: 0.7264 (KNN)
# AUC: 0.6372 (SVC)
# AUC: 0.7709 (CART)
# AUC: 0.8665 (RF)
# AUC: 0.736 (Adaboost)
# AUC: 0.7878 (GBM)
# AUC: 0.9007 (XGBoost)
# AUC: 0.8746 (LightGBM)
# AUC: 0.8645 (CatBoost)

# Basemodele başarılarına göre hiperparametre optimizasyonu kısmında ilerleyeceğimiz modeller:
# RF, XGBoost, LGBM, CatBoost

###################
# Automated Hyperparameter Optimization
###################


rf_params = {"max_features": [ 7, "auto"],
             "min_samples_split": [8, 15],
             "n_estimators": [500, 750]}

xgboost_params = {"learning_rate": [0.1, 0.001],
                  "n_estimators": [100, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}


lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [ 500, 750],
                   "colsample_bytree": [0.5, 0.7]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

classifiers = [("XGBoost", XGBClassifier(eval_metric='mlogloss'),xgboost_params),
               ("LightGBM", LGBMClassifier(), lightgbm_params),
               ("RF", RandomForestClassifier(), rf_params),
               ("CatBoost", CatBoostClassifier(verbose=False), catboost_params)]

best_models = {}

for name, classifier, params in classifiers:
    print(f"########## {name} ##########")
    cv_results = cross_validate(classifier, X_train, y_train, cv=5, n_jobs=-1,scoring=["roc_auc"])
    print(f"AUC (Before): {round(cv_results['test_roc_auc'].mean(),4)}")


    gs_best = GridSearchCV(classifier, params, cv=5, verbose=False,n_jobs=-1).fit(X_train,  y_train.values.ravel())
    final_model = classifier.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X_train,  y_train, cv=5, n_jobs=-1, scoring=["roc_auc"])
    print(f"AUC (After): {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


# ########## XGBoost ##########
# AUC (Before): 0.9007
# AUC (After): 0.9367
# XGBoost best params: {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'n_estimators': 1000}

# ########## LightGBM ##########
# AUC (Before): 0.8746
# AUC (After): 0.925
# LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'n_estimators': 750}

# ########## RF ##########
# AUC (Before): 0.8668
# AUC (After): 0.8813
# RF best params: {'max_features': 7, 'min_samples_split': 8, 'n_estimators': 750}

# ########## CatBoost ##########
# AUC (Before): 0.8645
# AUC (After): 0.8813
# CatBoost best params: {'depth': 6, 'iterations': 500, 'learning_rate': 0.1}


best_models
# {'XGBoost': XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
#                colsample_bynode=None, colsample_bytree=0.7,
#                enable_categorical=False, eval_metric='mlogloss', gamma=None,
#                gpu_id=None, importance_type=None, interaction_constraints=None,
#                learning_rate=0.1, max_delta_step=None, max_depth=None,
#                min_child_weight=None, missing=nan, monotone_constraints=None,
#                n_estimators=1000, n_jobs=None, num_parallel_tree=None,
#                predictor=None, random_state=None, reg_alpha=None,
#                reg_lambda=None, scale_pos_weight=None, subsample=None,
#                tree_method=None, validate_parameters=None, verbosity=None),
#  'LightGBM': LGBMClassifier(colsample_bytree=0.7, n_estimators=750),
#  'RF': RandomForestClassifier(max_features=7, min_samples_split=8, n_estimators=750),
#  'CatBoost': <catboost.core.CatBoostClassifier at 0x21b5778ad90>}


# Hybrid Feature Selection

# Feature Importances of 5 Models (CART,Random Forest, XGBoost, LightGBM, CatBoost)
feature_imp_all = pd.DataFrame({'Model': np.NaN,
                                'AUC_Score': np.NaN,
                                'Feature': np.NaN,
                                'Value': np.NaN,
                                'Weighted_Score': np.NaN}, index=[0])

for model, classifier in best_models.items():
    final_model = classifier.fit(X_train, y_train.values.ravel())
    cv_results = cross_validate(final_model, X_train, y_train, cv=5, scoring=["roc_auc"])
    score_final = round(cv_results['test_roc_auc'].mean(), 4)


    feature_imp = pd.DataFrame({'Model': model,
                                'AUC_Score': score_final * 100,
                                'Feature': X_train.columns,
                                'Value': final_model.feature_importances_,
                                'Weighted_Score': score_final * 100 * final_model.feature_importances_})

    feature_imp_all = pd.concat([feature_imp_all, feature_imp], axis=0)
    feature_imp_all.dropna(inplace=True)

feature_imp_all.sort_values("Weighted_Score", ascending=False).head()

#        Model  AUC_Score                 Feature    Value  Weighted_Score
# 11  LightGBM     92.500                       T 5588.000      516890.000
# 9   LightGBM     92.500  DiscountedItemSubTotal 4655.000      430587.500
# 3   LightGBM     92.500           IlkPesinFiyat 1892.000      175010.000
# 2   LightGBM     92.500              PesinFiyat 1799.000      166407.500
# 0   LightGBM     92.500    OrderDeliveryCityRef 1656.000      153180.000

feature_imp_all["Model"].unique()
# ['XGBoost', 'LightGBM', 'RF', 'CatBoost']

feature_imp_all.groupby("Model").agg({"AUC_Score": "mean"})

#           AUC_Score
# Model
# CatBoost     88.130
# LightGBM     92.500
# RF           88.080
# XGBoost      93.670

feature_imp_all = feature_imp_all.pivot_table(values = "Weighted_Score",
                                             columns="Model",
                                             index=['Feature'],
                                             aggfunc=np.mean).reset_index()

feature_imp_all.head()

# Model                 Feature  CatBoost   LightGBM     RF  XGBoost
# 0               CinsiyetKod_E    42.172  23957.500  0.598    2.425
# 1               CinsiyetKod_K    84.349  35520.000  0.745    3.589
# 2               CinsiyetKod_U    36.565   8695.000  0.491    3.826
# 3       DiscountAmountPerUnit    80.704  24605.000  1.304    3.712
# 4      DiscountedItemSubTotal  1275.491 430587.500 15.520    4.150

# Let's scale CV Score values for all models:

scaled_cols = [col for col in feature_imp_all.columns if feature_imp_all[col].dtypes == "float64"]

scaler = MinMaxScaler()
feature_imp_all[scaled_cols] = scaler.fit_transform(feature_imp_all[scaled_cols])

feature_imp_all.head()

# Model                 Feature  CatBoost  LightGBM    RF  XGBoost
# 0               CinsiyetKod_E     0.011     0.046 0.023    0.299
# 1               CinsiyetKod_K     0.022     0.069 0.029    0.442
# 2               CinsiyetKod_U     0.010     0.017 0.019    0.471
# 3       DiscountAmountPerUnit     0.021     0.048 0.051    0.457
# 4      DiscountedItemSubTotal     0.340     0.833 0.608    0.511

# Average Feature Importance based on Selected Models:

feature_imp_all["Avg_Importance_Value"] = ( (feature_imp_all["CatBoost"] +
                                           feature_imp_all["LightGBM"] +
                                           feature_imp_all["RF"] +
                                             feature_imp_all["XGBoost"]) /4 )* 100

feature_imp_all.sort_values("Avg_Importance_Value", ascending=False).head()

# Model                      Feature  CatBoost  LightGBM    RF  XGBoost  Avg_Importance_Value
# 22                               T     1.000     1.000 1.000    0.829                95.733
# 4           DiscountedItemSubTotal     0.340     0.833 0.608    0.511                57.308
# 16            OrderDeliveryCityRef     0.348     0.296 0.277    0.737                41.468
# 5                    IlkPesinFiyat     0.097     0.339 0.273    0.348                26.404
# 13     NEW_City_Segment_by_Sales_C     0.004     0.004 0.011    1.000                25.460


# Let's show the hybrid feature importance values with the plot:

plt.figure(figsize=(10, 10))
sns.set(font_scale=1)
sns.barplot(x="Avg_Importance_Value",
            y="Feature",
            data=feature_imp_all.sort_values(by="Avg_Importance_Value", ascending=False)[0:50])
plt.title('Hybrid Features Importance')
plt.tight_layout()
plt.show();
plt.savefig('features_importance.png')


selected_cols = list (feature_imp_all.loc[feature_imp_all["Avg_Importance_Value"] >=1, "Feature"].values)

selected_cols

# Let's observe the 5-Fold CV Scores of each models by using
# selected features which have been obtained with hybrid features selection method:

model_results_all = pd.DataFrame({'Model': np.NaN,
                                  'AUC_Score': np.NaN}, index=[0])

for model, classifier in best_models.items():
    final_model = classifier.fit(X_train[selected_cols], y_train.values.ravel())
    cv_results = cross_validate(final_model, X_train[selected_cols], y_train, cv=5, scoring=["roc_auc"])
    score_final = round(cv_results['test_roc_auc'].mean(), 4)

    model_results = pd.DataFrame({'Model': model,
                                  'AUC_Score': score_final}, index=[0])

    model_results_all = pd.concat([model_results_all, model_results], axis=0)
    model_results_all.dropna(inplace=True)

model_results_all.sort_values("AUC_Score", ascending=False).head(3)

#       Model  AUC_Score
# 0   XGBoost      0.936
# 0  LightGBM      0.924
# 0        RF      0.884

# Ensemble Learning

# Ensemble Modelling: XGBoost, LightGBM, Random Forest

voting_classifier_model = VotingClassifier(estimators= [('XGBoost',best_models['XGBoost']),
                                                        ('LightGBM',best_models['LightGBM']),
                                                        ('RF',best_models['RF'])],voting ='soft')

# Model Fit:
voting_classifier_model.fit(X_train[selected_cols], y_train)

# VotingClassifier(estimators=[('XGBoost',
#                               XGBClassifier(base_score=0.5, booster='gbtree',
#                                             colsample_bylevel=1,
#                                             colsample_bynode=1,
#                                             colsample_bytree=0.7,
#                                             enable_categorical=False,
#                                             eval_metric='mlogloss', gamma=0,
#                                             gpu_id=-1, importance_type=None,
#                                             interaction_constraints='',
#                                             learning_rate=0.1, max_delta_step=0,
#                                             max_depth=6, min_child_weight=1,
#                                             missing=nan,
#                                             monotone_con...
#                                             n_estimators=1000, n_jobs=8,
#                                             num_parallel_tree=1,
#                                             predictor='auto', random_state=0,
#                                             reg_alpha=0, reg_lambda=1,
#                                             scale_pos_weight=1, subsample=1,
#                                             tree_method='exact',
#                                             validate_parameters=1,
#                                             verbosity=None)),
#                              ('LightGBM',
#                               LGBMClassifier(colsample_bytree=0.7,
#                                              n_estimators=750)),
#                              ('RF',
#                               RandomForestClassifier(max_features=7,
#                                                      min_samples_split=8,
#                                                      n_estimators=750))],
#                  voting='soft')



# Model Performance Metrics for Train Set

cv_results_train = cross_validate(voting_classifier_model, X_train[selected_cols], y_train, cv=5, scoring=["accuracy", "f1", "roc_auc","precision","recall"])
cv_results_train

# {'fit_time': array([23.56229806, 23.18539548, 22.70827913, 23.59315276, 24.5592227 ]),
#  'score_time': array([1.45161557, 1.30443835, 1.32494426, 1.52818966, 1.30443144]),
#  'test_accuracy': array([0.87056312, 0.87696824, 0.86549239, 0.86789432, 0.86652429]),
#  'test_f1': array([0.75615887, 0.76515537, 0.74493927, 0.74987367, 0.74721941]),
#  'test_roc_auc': array([0.92873112, 0.93789105, 0.92787918, 0.93382807, 0.93000475]),
#  'test_precision': array([0.83835006, 0.86222732, 0.83257919, 0.83747178, 0.83408578]),
#  'test_recall': array([0.68864469, 0.68772894, 0.67399267, 0.67886551, 0.67673993])}



print("AUC Score for Train Set:", cv_results_train['test_roc_auc'].mean())

print("Accuracy Score for Train Set:", cv_results_train['test_accuracy'].mean())

print("F1 Score for Train Set:", cv_results_train['test_f1'].mean())

print("Precision Score for Train Set:", cv_results_train['test_precision'].mean())

print("Recall Score for Train Set:", cv_results_train['test_recall'].mean())

# AUC Score for Train Set: 0.9316668340020898
# Accuracy Score for Train Set: 0.8694884720732627
# F1 Score for Train Set: 0.7526693213218884
# Precision Score for Train Set: 0.8409428256504728
# Recall Score for Train Set: 0.6811943469765976

# Model Performance Metrics for Test Set

cv_results_test = cross_validate(voting_classifier_model, X_test[selected_cols], y_test, cv=5, scoring=["accuracy", "f1", "roc_auc","precision","recall"])

print("AUC Score for Test Set:", cv_results_test['test_roc_auc'].mean())

print("Accuracy Score for Test Set:", cv_results_test['test_accuracy'].mean())

print("F1 Score for Test Set:", cv_results_test['test_f1'].mean())

print("Precision Score for Test Set:", cv_results_test['test_precision'].mean())

print("Recall Score for Test Set:", cv_results_test['test_recall'].mean())

# AUC Score for Test Set: 0.8705529745512639
# Accuracy Score for Test Set: 0.8141747263959529
# F1 Score for Test Set: 0.6376128692379464
# Precision Score for Test Set: 0.7275394010119571
# Recall Score for Test Set: 0.5677260104905894


# * Confusion Matrix:

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    sns.set(rc = {'figure.figsize':(3, 3)})
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

y_pred = voting_classifier_model.predict(X_test[selected_cols])
plot_confusion_matrix(y_test, y_pred)


# Now, let's observe churn probabilites based calculated with Ensemble Modelling:


y_prob = voting_classifier_model.predict_proba(X_test[selected_cols])
y_prob = pd.DataFrame(y_prob ,index= X_test.index)
y_prob[0:5]

#           0     1
# 25597 0.950 0.050
# 4020  0.901 0.099
# 27    0.467 0.533
# 18845 0.989 0.011
# 23734 0.912 0.088



# 0.5 Eşik değerine göre churn olarak tahmin yapalım kalanlar 0 yani churn olmayanlar olsun.

y_prob["New_Churn_Value"] = y_prob[1].apply(lambda x: 1 if x > 0.50 else 0)
y_prob.head()

#           0     1  New_Churn_Value
# 25597 0.950 0.050                0
# 4020  0.901 0.099                0
# 27    0.467 0.533                1
# 18845 0.989 0.011                0
# 23734 0.912 0.088                0


# Test setimizi tahmin ettiğimiz churn değerleri ile birleştirelim.

y_test_ =  y_test.merge(y_prob, left_index=True, right_index=True)
y_test_[0:5]

#        Churn     0     1  New_Churn_Value
# 66384      1 0.289 0.711                1
# 69792      0 0.928 0.072                0
# 82237      0 0.872 0.128                0
# 59830      1 0.201 0.799                1
# 85627      0 0.980 0.020                0

# Test seti içerisinde tahmin ettiğimiz churn değerlerinin dağılımına bakalım.

y_test_.groupby("New_Churn_Value").agg({"New_Churn_Value": "count"})

# Ensemble learning ile oluşturduğumuz modeli tüm veri setine ekleyelim.

y_prob_all = voting_classifier_model.predict_proba(X[selected_cols])
y_prob_all = pd.DataFrame(y_prob_all ,index= X.index)

y_prob_all["New_Churn_Value"] = y_prob_all[1].apply(lambda x: 1 if x > 0.50 else 0)
y_prob_all[180:190]
#         0     1  New_Churn_Value
# 180 0.903 0.097                0
# 181 0.422 0.578                1
# 182 0.275 0.725                1
# 183 0.958 0.042                0
# 184 0.233 0.767                1
# 185 0.335 0.665                1
# 186 0.851 0.149                0
# 187 0.853 0.147                0
# 188 0.846 0.154                0
# 189 0.980 0.020                0

# Bu değerleri asıl veri seti ile birleştirelim.

y_ =  y.merge(y_prob_all, left_index=True, right_index=True)
y_.shape # (26763, 4)

target_df = pd.concat([df ,y_[["New_Churn_Value", 1]]],axis=1)
target_df.head()
target_df = target_df.rename(columns={1:"New_Churn_Prob"})



# Churn ihtimallerine göre segmentlere ayıralım.

target_df["New_Churn_Prob_Cat"] = np.where(target_df["New_Churn_Prob"] >=0.90,"Very High",np.NaN)
target_df["New_Churn_Prob_Cat"] = np.where(( (target_df["New_Churn_Prob"] <0.90) & (target_df["New_Churn_Prob"] >=0.80)),"High",target_df["New_Churn_Prob_Cat"])
target_df["New_Churn_Prob_Cat"] = np.where(( (target_df["New_Churn_Prob"] <0.80) & (target_df["New_Churn_Prob"] >=0.45)),"Medium",target_df["New_Churn_Prob_Cat"])
target_df["New_Churn_Prob_Cat"] = np.where(( (target_df["New_Churn_Prob"] <0.45) & (target_df["New_Churn_Prob"] >=0.10)),"Low",target_df["New_Churn_Prob_Cat"])
target_df["New_Churn_Prob_Cat"] = np.where(target_df["New_Churn_Prob"] <0.10,"Very_Low",target_df["New_Churn_Prob_Cat"])

summary_df = target_df.groupby("New_Churn_Prob_Cat").agg({"New_Churn_Prob": "mean",
                                                          "MusteriRef": "count"}).reset_index()
summary_df.sort_values("New_Churn_Prob", ascending=False)

#   New_Churn_Prob_Cat  New_Churn_Prob  MusteriRef
# 3          Very High           0.951        1199
# 0               High           0.847        1981
# 2             Medium           0.661        4435
# 1                Low           0.210        7492
# 4           Very_Low           0.042       11656

summary_df.rename(columns={'MusteriRef': 'Number_of_Customers'}, inplace=True)
summary_df.head()


# Görselleştirelim.

sns.barplot(y= summary_df["Number_of_Customers"], x = 'New_Churn_Prob_Cat', data = summary_df)
sns.set(rc = {'figure.figsize':(8, 6)})
plt.xlabel('Churn Probability Segments', fontsize=18)
plt.ylabel('Number of Customers', fontsize=16)
plt.show();
plt.savefig('last_segments_unbalanced.png')


# Veri setini kayıt edelim ve problem olursa tekrar bu modeli çalıştırmak zorunda kalmayalım.
path = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\CHURN_Prediction\\Final_df_unbalanced.csv "
target_df.to_csv(path, index=False,index_label=False)

df = pd.read_csv("Proje/Studies/Studies w_Datav2/CHURN_Prediction/Final_df_unbalanced.csv")

df.groupby("New_Churn_Prob_Cat").agg({"UnitPrice":"sum"})