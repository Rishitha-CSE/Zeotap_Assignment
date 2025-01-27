import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
customers = pd.read_csv("C:/Users/jrish/Downloads/Customers.csv")
transactions = pd.read_csv("C:/Users/jrish/Downloads/Transactions.csv")

# Merge data
merged_data = pd.merge(transactions, customers, on='CustomerID')

# Feature engineering
features = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionDate': 'count',
    # Add more features as needed
}).reset_index()

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features[['TotalValue', 'TransactionDate']])

# Clustering
db_indices = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    labels = kmeans.labels_
    db_index = davies_bouldin_score(scaled_features, labels)
    db_indices.append(db_index)

# Plot DB Index
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), db_indices, marker='o')
plt.title('Davies-Bouldin Index for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('DB Index')
plt.xticks(range(2, 11))
plt.grid()
plt.show()

# Final Clustering with optimal number of clusters (e.g., 4)
optimal_clusters = 4
kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=42)
features['Cluster'] = kmeans_final.fit_predict(scaled_features)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=features['TotalValue'], y=features['TransactionDate'], hue=features['Cluster'], palette='viridis')
plt.title('Customer Segmentation Clusters')
plt.xlabel('Total Transaction Value')
plt.ylabel('Number of Transactions')
plt.legend(title='Cluster')
plt.show()
