import pandas as pd
# Load the datasets
customers = pd.read_csv("C:/Users/jrish/Downloads/Customers.csv")
products = pd.read_csv("C:/Users/jrish/Downloads/Products.csv")
transactions = pd.read_csv("C:/Users/jrish/Downloads/Transactions.csv")

# Display the first few rows of each dataset
print(customers.head())
print(products.head())
print(transactions.head())

# Check for missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Plot total transactions over time
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
transactions.set_index('TransactionDate', inplace=True)
transactions.resample('M').size().plot(title='Monthly Transactions', figsize=(12, 6))
plt.ylabel('Number of Transactions')
plt.show()

# Plot total sales by product category
category_sales = transactions.groupby(transactions['ProductID'].map(products.set_index('ProductID')['Category'])).sum()['TotalValue']
category_sales.plot(kind='bar', title='Total Sales by Product Category', figsize=(12, 6))
plt.ylabel('Total Sales (USD)')
plt.show()
