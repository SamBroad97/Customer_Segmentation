# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:59:07 2026

@author: Sam
"""

# %% 1. Imports & Setup

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %% 2. Load Data & inital checks

df = pd.read_csv(r"archive/synthetic_customers.csv")


"""
checking data quality if anything needs cleaning or changing
"""
df.head()
df.columns
df.info()
df.isna().sum()

df.describe(include='all')
#checking if one row per customer
df.duplicated(subset='customer_id').sum()
df.dtypes

# %% 3. Fix dates & quality checks

#change date to datetime type
df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

#checking data quality
print(df[df['last_purchase_date'] < df['first_purchase_date']])
lastdate = df['last_purchase_date'].max()
firstdate = df['first_purchase_date'].min()

# %% 4. Explore a single customer

#checking biggest customer to see how data works, eg. loyalty points work e.g per order or cumulative 
biggest_id = df['customer_id'].value_counts().idxmax()  
bigcustomer_orders = df.loc[df['customer_id'] == biggest_id]  
bigcustomer_orders = df.loc[df['customer_id'] == biggest_id, ['last_purchase_date', 'loyalty_points', 'total_spent']]
bigcustomer_orders = bigcustomer_orders.sort_values('last_purchase_date', ascending=True)

# %% 5. Aggregate to customer level as currently transaction





#sort data by most recent pruchase
df_sorted = df.sort_values('last_purchase_date', ascending=False)


##aggregate data for analysis with fields needed so one row per customer
customer_summary = df_sorted.groupby('customer_id').agg(
    total_spent=('total_spent', 'sum'),
    loyalty_points=('loyalty_points', 'sum'),
    support_tickets=('support_tickets', 'mean'),
    age=('age', 'first'),
    income=('income', 'first'),
    subscription=('subscription', 'first'),
    churn_risk=('churn_risk', 'mean'),
    first_purchase_date=('first_purchase_date', 'min'),
    last_purchase_date=('last_purchase_date', 'max'),
    number_of_orders=('total_spent', 'count') 
).reset_index()


#checks for custom number of orders field i created
count = df_sorted['customer_id'] == biggest_id
raw_count = count.shape[0]
summary_count = customer_summary.loc[
    customer_summary['customer_id'] == biggest_id, 'number_of_orders'
].values[0]


#create new average order value for each customer
customer_summary['avg_order_value'] = customer_summary['total_spent']/customer_summary['number_of_orders']



customer_summary.head()

# %% 6. Create Recency (when customer last purchased) and Tenure (how long been customre)



reference_date = customer_summary['last_purchase_date'].max() #as static data using fixed date (latest purchase date)

customer_summary['recency'] = (
    reference_date - customer_summary['last_purchase_date']
).dt.days

customer_summary['tenure'] = (
    reference_date - customer_summary['first_purchase_date']
).dt.days

# %% 7. RFM Scoring for clusters and dashboard

"""
each customer is assigned a score from 1 to 5 for each dimension
using quantiles (1 = lowest group, 5 = highest group).
"""

customer_summary['R_score'] = pd.qcut(customer_summary['recency']
    ,5, labels =[5,4,3,2,1]).astype(int)
                                      
#most values 1-2  for no. of orders so have to do workaround using custom bins due to skewed data
customer_summary['F_score'] = pd.cut(customer_summary['number_of_orders'],
    bins=[0, 1, 2, 3, 5, 10],
    labels=[1, 2, 3, 4, 5],
    include_lowest=True).astype(int)


customer_summary['M_score'] = pd.qcut(customer_summary['total_spent']
    ,5,labels = [1,2,3,4,5]).astype(int)

#double checking breakdown
customer_summary['R_score'].value_counts().sort_index()
customer_summary['F_score'].value_counts().sort_index()
customer_summary['M_score'].value_counts().sort_index()
                          

customer_summary['RFM_score'] = (customer_summary['R_score']+customer_summary['F_score']+customer_summary['M_score'])       

#check final df
customer_summary.head()
customer_summary.info()

customer_summary['R_score'].value_counts().sort_index()
customer_summary['F_score'].value_counts().sort_index()
customer_summary['M_score'].value_counts().sort_index()


freq_bins = pd.qcut(
    customer_summary['number_of_orders'],
    q=5,
    duplicates='drop'
)

freq_bins.value_counts().sort_index()

# %% 8. Scaling & Silhouette scores



features = [
    'total_spent',
    'avg_order_value',
    'loyalty_points',
    'number_of_orders',
    'support_tickets',
    'recency',
    'tenure',
    'age',
    'income',
    'R_score',
    'F_score',
    'M_score'
]


x = customer_summary[features]

#create scaling
scaler = StandardScaler()

#fit and transformm
x_scaled = scaler.fit_transform(x)

#testing no. of diff clusters to measure silhouette score of each one
scores = {}

for k in range(2,10):
    km = KMeans(n_clusters = k, random_state = 36)
    km.fit(x_scaled)
    labels = km.labels_
    score = silhouette_score(x_scaled, labels)
    scores[k] = score
    
scores

# %% 9. Final KMeans model 


#creates model using 4 clusters as 2 best value but 4 may give more isight
kmeans = KMeans(n_clusters = 4, random_state = 36)
#fit on scaleddata
kmeans.fit(x_scaled)
#attach labels
customer_summary['cluster'] = kmeans.labels_

customer_summary.groupby('cluster')[features].mean()
df = customer_summary.groupby('cluster')[features].mean()
 
# %% 10. Cluster profiles & importance table

#get cluster means and overall dataset means to helo interpet what they are showing
cluster_means = customer_summary.groupby('cluster')[features].mean()
overall_means = customer_summary[features].mean()

importance = cluster_means - overall_means


# %% 11. Export

# Export the  customer-level dataset with all engineered features and cluster labels

customer_summary.to_csv("customer_segments.csv", index=False)













