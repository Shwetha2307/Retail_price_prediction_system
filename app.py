import matplotlib
matplotlib.use('Agg')  # Required for cloud deployment (Render)
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import io
import base64

app = Flask(__name__)

def get_cleaned_data():
    """Handles data ingestion and cleaning for the analysis engine."""
    try:
        df = pd.read_csv("Price_data.csv")
        df.columns = df.columns.str.replace(',', '').str.strip()
        df["Product"] = df["Product"].str.strip()
        df["Platform"] = df["Platform"].str.strip()
        df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        return df.dropna(subset=["Price", "Date", "Product", "Platform"])
    except Exception as e:
        print(f"File Error: {e}")
        return pd.DataFrame()

@app.route("/", methods=["GET", "POST"])
def home():
    df = get_cleaned_data()
    if df.empty:
        return "Error: Price_data.csv not found or empty. Please check the file."
    
    products = sorted(df["Product"].unique())
    result = None

    if request.method == "POST":
        product_name = request.form.get("product")
        df_product = df[df["Product"] == product_name].sort_values("Date")
        
        amazon_df = df_product[df_product["Platform"] == "Amazon"]
        flipkart_df = df_product[df_product["Platform"] == "Flipkart"]

        # Current Price Extraction
        amz_now = int(amazon_df["Price"].iloc[-1]) if not amazon_df.empty else 0
        flp_now = int(flipkart_df["Price"].iloc[-1]) if not flipkart_df.empty else 0

        # Percentage Math
        percent_diff = 0
        cheaper_platform = "N/A"
        if amz_now > 0 and flp_now > 0:
            diff = abs(amz_now - flp_now)
            percent_diff = round((diff / min(amz_now, flp_now)) * 100, 1)
            cheaper_platform = "Amazon" if amz_now < flp_now else "Flipkart"

        # AI Predictive Modeling
        predict_df = flipkart_df.copy() if len(flipkart_df) >= len(amazon_df) else amazon_df.copy()
        predicted_val = "N/A"
        recommendation = "NEUTRAL"
        insight = "Insufficient history for AI prediction."
        
        if len(predict_df) > 1:
            # FIX: Create Ordinal mapping BEFORE using it in X
            predict_df["Ordinal"] = predict_df["Date"].map(pd.Timestamp.toordinal)
            X = predict_df[["Ordinal"]].values 
            y = predict_df["Price"].values
            
            model = LinearRegression().fit(X, y)
            next_day_ordinal = predict_df["Ordinal"].max() + 1
            predicted_val = int(model.predict(np.array([[next_day_ordinal]]))[0])
            
            if predicted_val < predict_df["Price"].iloc[-1]:
                recommendation = "WAIT"
                insight = "Prices are predicted to drop tomorrow. Patience pays off!"
            else:
                recommendation = "BUY NOW"
                insight = "Prices are likely to rise. Grab the deal before it's gone!"

        # Visual Intelligence (Graph)
        plt.figure(figsize=(9, 4.5), facecolor='#1e293b')
        ax = plt.gca()
        ax.set_facecolor('#0f172a')
        if not amazon_df.empty:
            plt.plot(amazon_df["Date"], amazon_df["Price"], marker='o', color='#ff9900', label="Amazon", linewidth=2.5)
        if not flipkart_df.empty:
            plt.plot(flipkart_df["Date"], flipkart_df["Price"], marker='o', color='#cba6f7', label="Flipkart", linewidth=2.5)
        
        plt.title(f"Market Trend: {product_name}", color='white', pad=20)
        plt.xticks(rotation=25, color='#94a3b8')
        plt.yticks(color='#94a3b8')
        plt.grid(True, color='#334155', linestyle=':', alpha=0.4)
        plt.legend()
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png', facecolor='#1e293b')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        result = {
            "product": product_name, "amazon": amz_now, "flipkart": flp_now,
            "best_today": cheaper_platform, "percent": percent_diff,
            "predicted": predicted_val, "recommendation": recommendation,
            "insight": insight, "plot_url": plot_url
        }

    return render_template("index.html", products=products, result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)