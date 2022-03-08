###############################################################
# LC Waikiki Datası ile CLTV
###############################################################

# Daha önceden RFM eklenmiş data üzerinden ilerleyeceğiz.

import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Proje/Studies/Studies w_Datav2/RFM/RFM_LCW.csv")
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


# Müşteriref değişkeni grubunda OlusturmaTarihi değişkenin adet sayısı, Quantity değişkeninin toplammı ve LineTotal değişkeninin toplamını farklı bir df olarak kayıt edelim.


cltv_df = df.groupby("MusteriRef").agg({"OrderId": lambda x: x.nunique(),
                                        "Quantity" : lambda x:x.sum(),
                                        "UnitPrice": lambda x:x.sum()})

cltv_df.head()

cltv_df.columns=["Total_Transaction", "Total_Unit", "Total_Price"]

# Average Order Value Hesaplayalım

cltv_df["Avg_Order_Value"] = cltv_df["Total_Price"]/cltv_df["Total_Transaction"]

# Purchase Frequency Hesaplayalım.

cltv_df["Purchase_Frequency"] = cltv_df["Total_Transaction"] / cltv_df.shape[0]

# Repeat Rate ve Churn Rate'i Hesaplayalım

repeat_rate = cltv_df[cltv_df.Total_Transaction > 1].shape[0] / cltv_df.shape[0]
churn_rate = 1 - repeat_rate

# Kar miktarını alalım.

"""
Serap iş bilgisi ile kar oranının min %38 olması gerektiğini söyledi.
Bu sebeple kar oranı %38 alındı.
"""

cltv_df["Profit_Margin"] = cltv_df["Total_Price"] * 0.38

# Customer Value Hesaplayalım.

cltv_df["Customer_Value"] = (cltv_df["Avg_Order_Value"] * cltv_df["Purchase_Frequency"]) / churn_rate

# CLTV Hesaplayalım.

cltv_df["CLTV"] = cltv_df["Customer_Value"] * cltv_df["Profit_Margin"]

cltv_df.head()

# df ile CLTV'yi birleştirelim. Böylelikle model kuracağımız zaman elimiz kuvvetlenmiş olacak. Oluşturduğumuz yeni dataframe'i kayıt edelim.

df_merge = pd.merge(df, cltv_df, how="left", left_on="MusteriRef", right_on="MusteriRef")

# Verimizin son halini csv olarak kayıt edelim.

path = "C:\\Users\\omerkucuk\\Downloads\\Data Science ML Bootcamp\\Proje\\Studies\\Studies w_Datav2\\CLTV\\CLTV_LCW.csv "
df_merge.to_csv(path, index=False,index_label=False)