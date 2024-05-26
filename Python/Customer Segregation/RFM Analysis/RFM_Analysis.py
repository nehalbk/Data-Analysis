#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation using RFM Analysis
# This project aims to perform customer segmentation using RFM (Recency, Frequency, Monetary) analysis on a retail dataset. The segmentation will help in identifying different groups of customers based on their purchasing behavior.

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import time, warnings
import datetime as dt
warnings.filterwarnings("ignore")

# ## Load and Preview Data
# Load the dataset and preview the first few rows to understand its structure and data types.

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

# Display the first few rows of the dataset
data.head()

# ## Data Cleaning
# Remove unnecessary columns and handle missing values.

# Drop unnecessary columns and rows with missing values
clean_data = data.drop(columns=["InvoiceNo", "StockCode", "Description"]).dropna()

# Display summary statistics of the cleaned data
clean_data.describe()

# ## Exploratory Data Analysis
# Understand the distribution of the data and identify the most significant market.

# Display unique countries
print(clean_data["Country"].unique())

# Group by 'Country' and count the number of occurrences
grouped_data = clean_data.groupby('Country').count()

# Sort by 'Quantity' column in descending order
sorted_data = grouped_data.sort_values(by='Quantity', ascending=False)

# Display the top countries
sorted_data.head()

# ## Filter Data for United Kingdom
# Since the highest data comes from the UK, we will focus our analysis on the UK market.

# Filter data for 'United Kingdom'
uk_data = clean_data[clean_data["Country"] == "United Kingdom"]

# Drop the 'Country' column as it is no longer needed
uk_data.drop(columns=["Country"], inplace=True)

# Display the filtered data
print(uk_data.head())

# # Recency Analysis
# Recency indicates how recently a customer made a purchase.

# Group by CustomerID and aggregate by the maximum InvoiceDate to get the most recent purchase date
customer_recency_data = uk_data.groupby('CustomerID').agg({'InvoiceDate': 'max'})

# Rename the columns for clarity
customer_recency_data.columns = ["Recent InvoiceDate"]

# Reset the index to make CustomerID a column
customer_recency_data = customer_recency_data.reset_index()

# Display the first few rows
print(customer_recency_data.head())

# Calculate the "Recency" as the difference between now and the most recent InvoiceDate
customer_recency_data["Recency"] = dt.datetime.now() - customer_recency_data["Recent InvoiceDate"]

# Extract the number of days from the "Recency" timedelta
customer_recency_data["Recency"] = customer_recency_data["Recency"].dt.days

# Display the first few rows to verify
print(customer_recency_data.head())

# Remove outliers using the interquartile range (IQR)
Q1 = customer_recency_data["Recency"].quantile(0.25)
Q3 = customer_recency_data["Recency"].quantile(0.75)
IQR = Q3 - Q1
customer_recency_data = customer_recency_data[(customer_recency_data["Recency"] >= Q1 - 1.5 * IQR) & (customer_recency_data["Recency"] <= Q3 + 1.5 * IQR)]

# Display the filtered data
print(customer_recency_data.head())

# Normalize the "Recency" column using min-max normalization
min_recency = customer_recency_data["Recency"].min()
max_recency = customer_recency_data["Recency"].max()
customer_recency_data["Recency_normalized"] = customer_recency_data["Recency"].apply(lambda x: (x - min_recency) / (max_recency - min_recency))

# Reverse scale as lowest difference should be the highest value
customer_recency_data["Recency_normalized"] = 1 - customer_recency_data["Recency_normalized"]

# Create a distribution plot for Recency_normalized
sns.displot(customer_recency_data["Recency_normalized"], kde=True)

# # Frequency Analysis
# Frequency indicates how often a customer makes a purchase.

# Group by CustomerID and count the number of purchases to get the frequency
customer_freq_data = uk_data.groupby('CustomerID').size().reset_index(name='Frequency')

# Display the first few rows
print(customer_freq_data.head())

# Remove outliers using the interquartile range (IQR)
Q1 = customer_freq_data["Frequency"].quantile(0.25)
Q3 = customer_freq_data["Frequency"].quantile(0.75)
IQR = Q3 - Q1
customer_freq_data = customer_freq_data[(customer_freq_data["Frequency"] >= Q1 - 1.5 * IQR) & (customer_freq_data["Frequency"] <= Q3 + 1.5 * IQR)]

# Display the filtered data
print(customer_freq_data.head())

# Normalize the "Frequency" column using min-max normalization
min_freq = customer_freq_data["Frequency"].min()
max_freq = customer_freq_data["Frequency"].max()
customer_freq_data["Frequency_normalized"] = customer_freq_data["Frequency"].apply(lambda x: (x - min_freq) / (max_freq - min_freq))

# Display the first few rows to verify
print(customer_freq_data.head())

# Create a distribution plot for Frequency_normalized
sns.displot(customer_freq_data["Frequency_normalized"], kde=True)

# # Monetary Value Analysis
# Monetary value indicates how much money a customer spends.

# Calculate the total price for each purchase
uk_data["Total Price"] = uk_data["UnitPrice"] * uk_data["Quantity"]

# Group by CustomerID and sum the total price to get the monetary value
customer_monetary_data = uk_data.groupby('CustomerID').agg({'Total Price': 'sum'})

# Rename the columns for clarity
customer_monetary_data.columns = ["Monetary"]

# Reset the index to make CustomerID a column
customer_monetary_data = customer_monetary_data.reset_index()

# Display the first few rows
print(customer_monetary_data.head())

# Remove outliers using the interquartile range (IQR)
Q1 = customer_monetary_data["Monetary"].quantile(0.25)
Q3 = customer_monetary_data["Monetary"].quantile(0.75)
IQR = Q3 - Q1
customer_monetary_data = customer_monetary_data[(customer_monetary_data["Monetary"] >= Q1 - 1.5 * IQR) & (customer_monetary_data["Monetary"] <= Q3 + 1.5 * IQR)]

# Display the filtered data
print(customer_monetary_data.head())

# Normalize the "Monetary" column using min-max normalization
min_mon = customer_monetary_data["Monetary"].min()
max_mon = customer_monetary_data["Monetary"].max()
customer_monetary_data["Monetary_normalized"] = customer_monetary_data["Monetary"].apply(lambda x: (x - min_mon) / (max_mon - min_mon))

# Display the first few rows to verify
print(customer_monetary_data.head())

# Create a distribution plot for Monetary_normalized
sns.displot(customer_monetary_data["Monetary_normalized"], kde=True)

# # RFM Analysis with K-Means Clustering
# Combine the normalized RFM components and perform clustering.

# Combine the normalized columns into a new DataFrame
rfm_data = pd.concat([customer_recency_data["Recency_normalized"], customer_freq_data["Frequency_normalized"], customer_monetary_data["Monetary_normalized"]], axis=1)

# Display the first few rows of the new DataFrame
print(rfm_data.head())

# Remove NaN values as they were excluded during individual IQR analysis
rfm_data.dropna(inplace=True)

# # Determine the Optimal Number of Clusters using the Elbow Method

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

# From the Elbow plot, we determine that K=4 is the optimal number of clusters.

# # Perform K-Means Clustering

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

# # Analyze the Clusters

# Group by Cluster and calculate the average of each RFM component
rfm_data_cluster_avg = rfm_data.groupby('Cluster').agg({
    'Recency_normalized': 'mean',
    'Frequency_normalized': 'mean',
    'Monetary_normalized': 'mean'
})

# Display the result
print(rfm_data_cluster_avg)

# # Classify Clusters into Customer Segments
# Based on the RFM values, classify the clusters into different customer segments.
# - True Friends: High Recency, High Frequency, High Monetary
# - Butterflies: High Recency, Low Frequency, High Monetary
# - Barnacles: High Recency, High Frequency, Low Monetary
# - Strangers: Low Recency, Low Frequency, Low Monetary

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

# Display the result
print(rfm_data_cluster_avg)

# # Visualize the Clusters

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
