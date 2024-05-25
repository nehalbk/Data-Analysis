#!/usr/bin/env python
# coding: utf-8

# #Customer Segreggation using RFM Analysis

# In[350]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
#%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import time, warnings
import datetime as dt
warnings.filterwarnings("ignore")


# In[351]:


# Load the data with specified encoding and data types
data = pd.read_csv(
    './data.csv',
    encoding="ISO-8859-1",
    dtype={
        'CustomerID': str,
        'InvoiceNo': str,
        'StockCode': str,
        'Description': str,
        'Quantity': int,
        'UnitPrice': float,
        'Country': str
    },
    parse_dates=['InvoiceDate']
)

# Display the data types to verify
print(data.dtypes)


# In[352]:


data.head()


# In[353]:


data.describe()


# In[354]:


clean_data=data.drop(columns=["InvoiceNo","StockCode","Description"]).dropna()


# In[355]:


clean_data.describe()


# In[356]:


clean_data["Country"].unique()


# In[357]:


# Group by 'Country' and count the number of occurrences
grouped_data = clean_data.groupby('Country').count()

# Sort by 'Quantity' column in descending order
sorted_data = grouped_data.sort_values(by='Quantity', ascending=False)


# In[358]:


sorted_data.head()


# In[359]:


#Filter data for 'United Kingdom'
uk_data = clean_data[clean_data["Country"] == "United Kingdom"]

uk_data.drop(inplace=True,columns=["Country"])

# Display the filtered data
print(uk_data)


# # Recency

# In[360]:


# Grouping by CustomerID and aggregating by maximum InvoiceDate
customer_recency_data = uk_data.groupby('CustomerID').agg({'InvoiceDate': 'max'})

# Renaming the columns
customer_recency_data.columns = ["Recent InvoiceDate"]

# Resetting the index to make CustomerID a column
customer_recency_data = customer_recency_data.reset_index()

customer_recency_data.head()


# In[361]:


# Calculate the "Recency" as the difference between now and "InvoiceDate"
customer_recency_data["Recency"] = dt.datetime.now() - customer_recency_data["Recent InvoiceDate"]

# Extract the number of days from the "Recency" timedelta
customer_recency_data["Recency"] = customer_recency_data["Recency"].dt.days

# Display the first few rows to verify
customer_recency_data.head()


# In[362]:


# Calculate the interquartile range (IQR) for Frequency_normalized
Q1 = customer_recency_data["Recency"].quantile(0.25)
Q3 = customer_recency_data["Recency"].quantile(0.75)
IQR = Q3 - Q1

# Filter out the data points within the IQR range
customer_recency_data = customer_recency_data[(customer_recency_data["Recency"] >= Q1 - 1.5 * IQR) & (customer_recency_data["Recency"] <= Q3 + 1.5 * IQR)]

# Display the filtered data
print(customer_freq_data.head())


# In[363]:


# Normalize the "Recency" column using min-max normalization

min_recency=customer_data["Recency"].min()
max_recency=customer_data["Recency"].max()

# Calculate min-max normalization using apply and lambda function
customer_data["Recency_normalized"] = customer_data["Recency"].apply(lambda x: (x - min_recency) / (max_recency - min_recency))

#Reverse sacled as lowest difference should be highest value
customer_data["Recency_normalized"]=1-customer_data["Recency_normalized"]


# In[364]:


# Create a distribution plot for Recency_normalized
sns.displot(customer_data["Recency_normalized"], kde=True)


# ## Frequency

# In[365]:


# Grouping by CustomerID and aggregating by count to get frequency
customer_freq_data = uk_data.groupby('CustomerID').size().reset_index(name='Frequency')

# Displaying the result
print(customer_freq_data.head())


# In[366]:


# Calculate the interquartile range (IQR) for Frequency_normalized
Q1 = customer_freq_data["Frequency"].quantile(0.25)
Q3 = customer_freq_data["Frequency"].quantile(0.75)
IQR = Q3 - Q1

# Filter out the data points within the IQR range
customer_freq_data = customer_freq_data[(customer_freq_data["Frequency"] >= Q1 - 1.5 * IQR) & (customer_freq_data["Frequency"] <= Q3 + 1.5 * IQR)]

# Display the filtered data
print(customer_freq_data.head())


# In[367]:


# Normalize the "Frequency" column using min-max normalization

min_freq=customer_freq_data["Frequency"].min()
max_freq=customer_freq_data["Frequency"].max()

# Calculate min-max normalization using apply and lambda function
customer_freq_data["Frequency_normalized"] = customer_freq_data["Frequency"].apply(lambda x: (x - min_freq) / (max_freq - min_freq))

print(customer_freq_data.head())


# In[368]:


# Create a distribution plot for Recency_normalized
sns.displot(customer_freq_data["Frequency_normalized"], kde=True)


# # Monetory Value

# In[369]:


# Grouping by CustomerID and aggregating by sum of the product of 'UnitPrice' and 'Quantity'
customer_monetary_data=[]
customer_monetary_data = uk_data[["CustomerID","UnitPrice","Quantity"]]

# Creating a new column for the product of 'UnitPrice' and 'Quantity'
customer_monetary_data['Total Price'] = customer_monetary_data['UnitPrice'] * customer_monetary_data['Quantity']

# Grouping by CustomerID and aggregating by sum Monetary
customer_monetary_data = customer_monetary_data.groupby('CustomerID').agg({'Total Price': 'sum'})

# Renaming the columns
customer_monetary_data.columns = ["Monetary"]

# Resetting the index to make CustomerID a column
customer_monetary_data = customer_monetary_data.reset_index()


# In[370]:


# Displaying the result
print(customer_monetary_data.head(10))


# In[371]:


# Calculate the interquartile range (IQR) for Frequency_normalized
Q1 = customer_monetary_data["Monetary"].quantile(0.25)
Q3 = customer_monetary_data["Monetary"].quantile(0.75)
IQR = Q3 - Q1

# Filter out the data points within the IQR range
customer_monetary_data = customer_monetary_data[(customer_monetary_data["Monetary"] >= Q1 - 1.5 * IQR) & (customer_monetary_data["Monetary"] <= Q3 + 1.5 * IQR)]

# Display the filtered data
print(customer_monetary_data.head())


# In[372]:


# Normalize the "Monetary" column using min-max normalization

min_mon=customer_monetary_data["Monetary"].min()
max_mon=customer_monetary_data["Monetary"].max()

# Calculate min-max normalization using apply and lambda function
customer_monetary_data["Monetary_normalized"] = customer_monetary_data["Monetary"].apply(lambda x: (x - min_mon) / (max_mon - min_mon))

print(customer_monetary_data.head())


# In[373]:


# Create a distribution plot for Recency_normalized
sns.displot(customer_monetary_data["Monetary_normalized"], kde=True)


# In[374]:


#RFM

# Combine the normalized columns into a new DataFrame
rfm_data = pd.concat([customer_data["Recency_normalized"], customer_freq_data["Frequency_normalized"], customer_monetary_data["Monetary_normalized"]], axis=1)

# Display the first few rows of the new DataFrame
print(rfm_data.head())


# In[375]:


rfm_data.dropna(inplace=True)
rfm_data


# In[380]:


# Assuming rfm_data is already defined and contains the necessary data

# Define the range of K values to test
k_range = range(1, 11)

# Initialize an empty list to store inertia values for each K
inertia_values = []

# Compute the inertia for each K
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_data)
    inertia_values.append(kmeans.inertia_)

# Plot the inertia values to find the elbow point
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia_values, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


# In[376]:


# Instantiate the KMeans model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model to the RFM data
kmeans.fit(rfm_data)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Add the cluster labels to the RFM data DataFrame
rfm_data['Cluster'] = cluster_labels

# Display the first few rows of the RFM data with cluster labels
print(rfm_data.head())


# In[377]:


# Grouping by Cluster and calculating the average of each RFM component
rfm_data_cluster_avg = rfm_data.groupby('Cluster').agg({
    'Recency_normalized': 'mean',
    'Frequency_normalized': 'mean',
    'Monetary_normalized': 'mean'
})

# Displaying the result
print(rfm_data_cluster_avg)


# In[378]:


#Renaming

# Renaming the clusters
cluster_names = {
    0: 'Butterflies',
    1: 'Strangers',
    2: 'Barnacles',
    3: 'True Friends'
}
rfm_data_cluster_avg.rename(index=cluster_names, inplace=True)

# Renaming the column names
column_names = {
    'Recency_normalized': 'Recency',
    'Frequency_normalized': 'Frequency',
    'Monetary_normalized': 'Monetary'
}
rfm_data_cluster_avg.rename(columns=column_names, inplace=True)

# Displaying the result
print(rfm_data_cluster_avg)


# In[379]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Reduce dimensionality using PCA
pca = PCA(n_components=3)
rfm_pca = pca.fit_transform(rfm_data)

# Plotting the clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
for cluster in range(4):
    ax.scatter(rfm_pca[rfm_data['Cluster'] == cluster][:, 0],
               rfm_pca[rfm_data['Cluster'] == cluster][:, 1],
               rfm_pca[rfm_data['Cluster'] == cluster][:, 2],
               label=cluster_names[cluster])

# Set labels and title
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('Clusters Visualization')

# Add legend
ax.legend()

# Show plot
plt.show()


# In[ ]:





# In[ ]:




