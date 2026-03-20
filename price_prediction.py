import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
df = pd.read_csv("Price_data.csv")

# This line is the magic fix: 
# It removes spaces AND any trailing commas from your header names
df.columns = df.columns.str.replace(',', '').str.strip()

print("\n--- Fixed Columns ---")
print(df.columns.tolist()) 

# -------------------------------
# STEP 2: CLEAN DATA
# -------------------------------
# Now "Price" (Capital P, no comma) will work!
df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True) # Added dayfirst for your DD-MM format

df = df.dropna(subset=["Price", "Date"])
df = df.sort_values(by="Date")

# -------------------------------
# STEP 3: PRICE COMPARISON
# -------------------------------
print("\n--- Latest Price Comparison ---")
latest_date = df["Date"].max()
latest_data = df[df["Date"] == latest_date]

if not latest_data.empty:
    print(latest_data[["Product", "Platform", "Price"]])
    # Use idxmin() safely
    best_idx = latest_data["Price"].idxmin()
    best = latest_data.loc[best_idx]
    print(f"\nBest Platform to Buy: {best['Platform']} at {best['Price']}")
else:
    print("No data found for the latest date.")

# -------------------------------
# STEP 4: VISUALIZATION
# -------------------------------
# Select a specific product safely
unique_products = df["Product"].unique()
if len(unique_products) > 0:
    product_name = unique_products[0]
    product_df = df[df["Product"] == product_name]

    amazon = product_df[product_df["Platform"].str.strip() == "Amazon"]
    flipkart = product_df[product_df["Platform"].str.strip() == "Flipkart"]

    plt.figure(figsize=(10, 5))
    plt.plot(amazon["Date"], amazon["Price"], marker='o', label="Amazon")
    plt.plot(flipkart["Date"], flipkart["Price"], marker='o', label="Flipkart")

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Price Trend for {product_name}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------------
# STEP 5: PREDICTION MODEL
# -------------------------------
# Fix: Using .copy() to avoid SettingWithCopyWarning
data = flipkart.copy()

if len(data) > 1:
    # Linear Regression needs numeric X. We use ordinal dates for better accuracy
    data["Date_Ordinal"] = data["Date"].map(pd.Timestamp.toordinal)
    
    X = data["Date_Ordinal"].values.reshape(-1, 1)
    y = data["Price"].values

    model = LinearRegression()
    model.fit(X, y)

    # Predict for "Tomorrow"
    next_day_timestamp = data["Date"].max() + pd.Timedelta(days=1)
    next_day_ordinal = np.array([[next_day_timestamp.toordinal()]])
    predicted_price = model.predict(next_day_ordinal)[0]

    print(f"\nPredicted Price for {next_day_timestamp.date()}: {round(predicted_price, 2)}")

    # -------------------------------
    # STEP 6: RECOMMENDATION
    # -------------------------------
    current_price = data["Price"].iloc[-1]
    
    if predicted_price < current_price:
        print("Recommendation: WAIT (Price may drop)")
    else:
        print("Recommendation: BUY NOW (Price may increase)")
else:
    print("\n[!] Not enough data points on Flipkart to make a prediction.")

# -------------------------------
# STEP 7: STABILITY & INSIGHTS
# -------------------------------
print("\n--- Price Stability (Volatility) ---")
stability = df.groupby("Platform")["Price"].std().fillna(0)
for platform, vol in stability.items():
    print(f"{platform} volatility: {round(vol, 2)}")

avg_price = df.groupby("Platform")["Price"].mean()
print(f"\nBest Platform Overall (Lowest Avg): {avg_price.idxmin()}")

# -------------------------------
# PRICE DIFFERENCE PERCENTAGE
# # -------------------------------
if len(latest_data) >= 2:
        prices = latest_data.set_index("Platform")["Price"]
        
if "Amazon" in prices and "Flipkart" in prices:
            amazon_price = prices["Amazon"]
            flipkart_price = prices["Flipkart"]
            
            price_diff = abs(amazon_price - flipkart_price)
            percent_diff = (price_diff / max(amazon_price, flipkart_price)) * 100
            
if amazon_price < flipkart_price:
                cheaper = "Amazon"
else:
                cheaper = "Flipkart"
            
print(f"Price Difference: ₹{price_diff}")
print(f"Percentage Difference: {round(percent_diff, 2)}%")
print(f"{cheaper} is cheaper by {round(percent_diff, 2)}%")