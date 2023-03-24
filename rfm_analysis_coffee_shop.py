##############################
# Import Data
##############################
import pandas as pd
import datetime as dt
def import_csv(dataframe):
    df = pd.read_csv(dataframe)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    pd.set_option('display.html.table_schema', True)
    return df
df_ = import_csv("/Users/busegungor/PycharmProjects/coffee_shop/coffee shop/201904 sales reciepts.csv")
df = df_.copy()

def describe(dataframe):
    print("###Shape###")
    print(dataframe.shape)
    print("###Types###")
    print(dataframe.dtypes)
    print("###Variables###")
    print(dataframe.columns)
    print("###NA###")
    print(dataframe.isnull().sum())
    print("###Describe###")
    print(dataframe.describe().T)
    print("###Quick View###")
    print(dataframe.head())

describe(df)
##############################
# Data Understanding
##############################
df["product_id"].nunique() # 80 different product
df.groupby("product_id").agg({"quantity": "sum"}).sort_values("quantity", ascending=False)
# how many orders were placed for each product?
#            quantity
#product_id
#50              1558
#59              1546
#38              1531
#54              1513
#32              1506
#              ...
#15                48
#4                 47
#12                45
#18                42
#19                36
#[80 rows x 1 columns]

df["transaction_id"].nunique() # how many invoices have been issued in total? 4203
df.groupby("transaction_id").agg({"line_item_amount" : "sum"}) # How much is earned per invoice?

#               line_item_amount
#transaction_id
#1                      283.25000
#2                      221.95000
#3                      314.65000
#4                      297.23000
#5                      354.55000
#                          ...
#4199                     4.50000
#4200                     4.75000
#4201                     6.00000
#4202                     2.50000
#4203                     3.00000
#[4203 rows x 1 columns]

##############################
# Calculating RFM Metrics
##############################

# When creating metrics for recency, it is done by taking a distance from a certain day.
# For this, datetime is used. We chose two after the last transaction date.

df["transaction_date"] = pd.to_datetime(df["transaction_date"]) # Change object to datetime
df["transaction_date"].max()
today_date = dt.datetime(2019, 5, 1)

# Recency: customer_id'ye göre groupby aldıktan sonra today_date'den ile transaction_date'in max'ını alıp kaç gün olduğu.
# Frequency: customer_id'ye göre groupby aldıktan sonra her bir müşterinin eşsiz transaction_id sayısına gidersek.
# Monetary: customer_id'ye göre groupby aldıktan sonra line_item_amount'ların sum'ını alırsak her bir müşterinin ne kadar para bıraktığını öğreniriz.
rfm = df.groupby("customer_id").agg({"transaction_date": lambda transaction_date: (today_date - transaction_date.max()).days,
                                     "transaction_id": lambda transaction_id: transaction_id.nunique(),
                                     "line_item_amount": lambda line_item_amount: line_item_amount.sum()})
rfm.columns = ["recency", "frequency", "monetary"]
rfm.describe().T

#               count      mean        std     min      25%      50%      75%          max
#recency   2248.00000   6.07162    5.24824 2.00000  3.00000  4.00000  7.00000     30.00000
#frequency 2248.00000  10.35632   76.99555 1.00000  6.00000  8.00000 11.00000   3655.00000
#monetary  2248.00000 103.93058 2515.50926 2.45000 33.22500 47.00000 65.25000 119312.72000


# Convert to 1-5 score with qcut() function to recency, frequency and monetary metrics.
rfm["recency_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"], 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# Express RF_Score with recency and frequency
rfm["RFM_Score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# This rfm score express with regular expression.
# A regular expression (shortened as regex or regexp; sometimes referred to as rational expression) is a sequence of
# characters that specifies a match pattern in text. Usually such patterns are used by string-searching algorithms for
# "find" or "find and replace" operations on strings, or for input validation.
# https://en.wikipedia.org/wiki/Regular_expression#:~:text=A%20regular%20expression%20(shortened%20as,strings%2C%20or%20for%20input%20validation.
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
rfm["segment"] = rfm["RFM_Score"].replace(seg_map, regex=True)

# According to profile of customer analysis recency, frequency, monetary
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#                         recency       frequency        monetary
#                        mean count      mean count      mean count
#segment
#about_to_sleep       4.41958   143   5.54545   143  35.06049   143
#at_Risk              8.85303   347   9.75793   347  55.82455   347
#cant_loose           7.24272   103  15.05825   103  88.75680   103
#champions            2.00000   137  38.61314   137 940.76715   137
#hibernating         12.97327   449   5.11804   449  29.91392   449
#loyal_customers      3.26456   412  13.07524   412  75.26784   412
#need_attention       4.30120    83   8.54217    83  51.52434    83
#new_customers        2.00000   140   5.15714   140  28.99393   140
#potential_loyalists  2.44713   331   7.95468   331  46.71435   331
#promising            2.99029   103   4.97087   103  28.97388   103

# save to csv
rfm.to_csv("rfm.csv")

# Let's say we don't want to lose "need attention" group. We can take this customers' customer_id from rfm dataframe and
# save to new dataframe and according to this customer organized a campaign.

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "need_attention"].index
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)
new_df.to_csv('new_customer.csv')
