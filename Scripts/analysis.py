## Task -1 Import and load the date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel("Datas/Online Retail.xlsx")

# Display the first 5 rows
print(df.head())
# Display the last 5 rows
print(df.tail())

# Display the number of rows and columns
print(df.shape)

# Display column names
print(df.columns)

# Display dataset information
df.info()
# Convert InvoiceDate into datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(df["InvoiceDate"].dtype)

## Task 2 - Data Cleaning

## Remove rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

print("Shape after removing missing CustomerID:")
print(df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

print("Shape after removing duplicates:")
print(df.shape)

# Remove rows with negative or zero Quantity
df = df[df["Quantity"] > 0]

print("Shape after removing invalid Quantity:")
print(df.shape)


# Remove rows with invalid UnitPrice
df = df[df["UnitPrice"] > 0]

print("Shape after removing invalid UnitPrice:")
print(df.shape)


## Task 3 – Feature Engineering

# Create TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print(df[["Quantity", "UnitPrice", "TotalPrice"]].head())

# Extract Year
df["Year"] = df["InvoiceDate"].dt.year

# Extract Month
df["Month"] = df["InvoiceDate"].dt.month

# Extract Day
df["Day"] = df["InvoiceDate"].dt.day

# Extract Hour
df["Hour"] = df["InvoiceDate"].dt.hour

print(df[["InvoiceDate", "Year", "Month", "Day", "Hour"]].head())

# Create Order Size category
df["Order Size"] = np.where(
    df["Quantity"] <= 5, "Small",
    np.where(df["Quantity"] <= 20, "Medium", "Large")
)

print(df["Order Size"].value_counts())

# Create Day Type
df["Day Type"] = np.where(
    df["InvoiceDate"].dt.day_of_week >= 5,
    "Weekend",
    "Weekday"
)

print(df["Day Type"].value_counts())


# Calculate total spending for each customer
customer_sales = df.groupby("CustomerID")["TotalPrice"].sum()

# Create Customer Segment
customer_segment = pd.cut(
    customer_sales,
    bins=[0, 500, 2000, float("inf")],
    labels=["Low Value", "Medium Value", "High Value"]
)

# Add Customer Segment to dataset
df["Customer Segment"] = df["CustomerID"].map(customer_segment)

print(df["Customer Segment"].value_counts())

# Task 4 - Data Exploration

# Display statistical summary
print("Statistical Summary:")
print(df.describe())

# Count transactions by country
print("Transactions by Country:")
print(df["Country"].value_counts())

# Display unique countries
print("Unique Countries:")
print(df["Country"].unique())

# Count different products
print("Number of Unique Products:")
print(df["StockCode"].nunique())

# Group sales by country
country_sales = df.groupby("Country")["TotalPrice"].sum()

# Sort countries by sales
country_sales = country_sales.sort_values(ascending=False)

print("Sales by Country:")
print(country_sales)

# Group sales by month
monthly_sales = df.groupby("Month")["TotalPrice"].sum()

print("Monthly Sales:")
print(monthly_sales)

# Group sales by product description
product_sales = df.groupby("Description")["TotalPrice"].sum()

# Sort products by sales
product_sales = product_sales.sort_values(ascending=False)

print("Top 10 Products:")
print(product_sales.head(10))

# Calculate total sales for each customer
customer_sales = df.groupby("CustomerID")["TotalPrice"].sum()

# Sort customers by total sales
customer_sales = customer_sales.sort_values(ascending=False)

print("Top 10 Customers:")
print(customer_sales.head(10))

# Task 5 - Data Wrangling

# Total sales by country
country_sales = df.groupby("Country")["TotalPrice"].sum()

# Top 5 countries by sales
top_5_countries = country_sales.sort_values(ascending=False).head(5)

print("Top 5 Countries by Sales:")
print(top_5_countries)

# Total sales by month
monthly_sales = df.groupby("Month")["TotalPrice"].sum()

# Best sales month
best_month = monthly_sales.idxmax()

print("Best Sales Month:")
print(best_month)

# Total sales by customer
customer_sales = df.groupby("CustomerID")["TotalPrice"].sum()

# Top 5 customers
top_5_customers = customer_sales.sort_values(ascending=False).head(5)

print("Top 5 Customers:")
print(top_5_customers)

# Sales by product
product_sales = df.groupby("Description")["TotalPrice"].sum()

# Top 5 products
top_5_products = product_sales.sort_values(ascending=False).head(5)

print("Top 5 Products:")
print(top_5_products)


# Task 6 - Statistical Analysis

# Mean
mean_quantity = df["Quantity"].mean()
mean_price = df["UnitPrice"].mean()
mean_sales = df["TotalPrice"].mean()

print("Mean Quantity:", mean_quantity)
print("Mean Unit Price:", mean_price)
print("Mean Total Sales:", mean_sales)

# Median
median_quantity = df["Quantity"].median()
median_price = df["UnitPrice"].median()
median_sales = df["TotalPrice"].median()

print("Median Quantity:", median_quantity)
print("Median Unit Price:", median_price)
print("Median Total Sales:", median_sales)

# Mode
mode_quantity = df["Quantity"].mode()[0]
mode_price = df["UnitPrice"].mode()[0]

print("Mode Quantity:", mode_quantity)
print("Mode Unit Price:", mode_price)

# Standard Deviation
std_quantity = df["Quantity"].std()
std_price = df["UnitPrice"].std()
std_sales = df["TotalPrice"].std()

print("Standard Deviation of Quantity:", std_quantity)
print("Standard Deviation of Unit Price:", std_price)
print("Standard Deviation of Total Sales:", std_sales)

# Variance
variance_quantity = df["Quantity"].var()
variance_price = df["UnitPrice"].var()
variance_sales = df["TotalPrice"].var()

print("Variance of Quantity:", variance_quantity)
print("Variance of Unit Price:", variance_price)
print("Variance of Total Sales:", variance_sales)

# Percentiles
print("Quantity Percentiles:")
print(df["Quantity"].quantile([0.25, 0.50, 0.75]))

print("Unit Price Percentiles:")
print(df["UnitPrice"].quantile([0.25, 0.50, 0.75]))

print("Total Sales Percentiles:")
print(df["TotalPrice"].quantile([0.25, 0.50, 0.75]))

# Task 7 - Data Visualization

import os

# Create Images folder if it does not exist
os.makedirs("Images", exist_ok=True)

# 1. Line Plot - Monthly Sales
monthly_sales = df.groupby("Month")["TotalPrice"].sum()

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.savefig("Images/monthly_sales.png")
plt.show()


# 2. Bar Chart - Top 10 Countries by Sales
country_sales = df.groupby("Country")["TotalPrice"].sum()
top_countries = country_sales.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_countries.plot(kind="bar")
plt.title("Top 10 Countries by Sales")
plt.xlabel("Country")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Images/top_countries.png")
plt.show()


# 3. Histogram - Total Price Distribution
plt.figure(figsize=(10, 5))
plt.hist(df["TotalPrice"], bins=50)
plt.title("Total Price Distribution")
plt.xlabel("Total Price")
plt.ylabel("Frequency")
plt.savefig("Images/total_price_histogram.png")
plt.show()


# 4. Boxplot - Unit Price
plt.figure(figsize=(10, 5))
plt.boxplot(df["UnitPrice"])
plt.title("Unit Price Boxplot")
plt.ylabel("Unit Price")
plt.savefig("Images/unit_price_boxplot.png")
plt.show()


# 5. Countplot - Day Type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Day Type")
plt.title("Transactions by Day Type")
plt.xlabel("Day Type")
plt.ylabel("Count")
plt.savefig("Images/day_type_countplot.png")
plt.show()


# 6. Violin Plot - Quantity by Order Size
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x="Order Size", y="Quantity")
plt.title("Quantity Distribution by Order Size")
plt.xlabel("Order Size")
plt.ylabel("Quantity")
plt.savefig("Images/order_size_violinplot.png")
plt.show()


# 7. Heatmap - Correlation Matrix
numeric_df = df[["Quantity", "UnitPrice", "TotalPrice", "Year", "Month", "Day", "Hour"]]

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("Images/correlation_heatmap.png")
plt.show()


# 8. Pairplot - Numerical Variables
sample_df = df[["Quantity", "UnitPrice", "TotalPrice"]].sample(1000, random_state=42)

sns.pairplot(sample_df)
plt.savefig("Images/pairplot.png")
plt.show()

# Task 8 - Business Insights

# 1. Country with highest sales
country_sales = df.groupby("Country")["TotalPrice"].sum()
highest_sales_country = country_sales.idxmax()

print("Country with Highest Sales:", highest_sales_country)
print("Sales:", country_sales.max())


# 2. Best sales month
monthly_sales = df.groupby("Month")["TotalPrice"].sum()
best_month = monthly_sales.idxmax()

print("Best Sales Month:", best_month)
print("Sales:", monthly_sales.max())

# 3. Peak sales hour
hourly_sales = df.groupby("Hour")["TotalPrice"].sum()
peak_hour = hourly_sales.idxmax()

print("Peak Sales Hour:", peak_hour)
print("Sales:", hourly_sales.max())


# 4. Customer behavior
average_quantity = df["Quantity"].mean()
average_sales = df["TotalPrice"].mean()

print("Average Quantity per Transaction:", average_quantity)
print("Average Sales per Transaction:", average_sales)


# 5. High-value customers
customer_sales = df.groupby("CustomerID")["TotalPrice"].sum()
top_customers = customer_sales.sort_values(ascending=False).head(5)

print("Top 5 High-Value Customers:")
print(top_customers)


# 6. Top products
product_sales = df.groupby("Description")["TotalPrice"].sum()
top_products = product_sales.sort_values(ascending=False).head(5)

print("Top 5 Products:")
print(top_products)