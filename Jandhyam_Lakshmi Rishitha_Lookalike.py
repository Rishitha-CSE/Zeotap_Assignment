import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load data
customers = pd.read_csv("C:/Users/jrish/Downloads/Customers.csv")
transactions = pd.read_csv("C:/Users/jrish/Downloads/Transactions.csv")
products = pd.read_csv("C:/Users/jrish/Downloads/Products.csv")

# Merge data
merged_data = pd.merge(transactions, customers, on='CustomerID')
merged_data = pd.merge(merged_data, products, on='ProductID')

# Feature engineering
features = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionDate': 'count',
    # Add more features as needed
}).reset_index()

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Calculate similarity
similarity_matrix = cosine_similarity(scaled_features)

# Generate recommendations
lookalikes = {}
for i in range(len(similarity_matrix)):
    similar_indices = similarity_matrix[i].argsort()[-4:-1][::-1]  # Top 3 lookalikes
    lookalikes[features['CustomerID'][i]] = [(features['CustomerID'][j], similarity_matrix[i][j]) for j in similar_indices]

# Create DataFrame for Lookalike.csv
lookalike_df = pd.DataFrame.from_dict(lookalikes, orient='index', columns=['lookalike_id_1', 'score_1', 'lookalike_id_2', 'score_2', 'lookalike_id_3', 'score_3'])
lookalike_df.to_csv('Lookalike.csv', index=True)