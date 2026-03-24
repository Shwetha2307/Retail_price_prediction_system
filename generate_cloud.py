import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import os

# 1. Load your dataset
df = pd.read_csv('Price_data.csv')

# 2. Combine all product names into one string
text = " ".join(item for item in df.Product.astype(str))

# 3. Define Stopwords (Standard + Your own "Boring" words)
comment_words = set(STOPWORDS)
# Add words you don't want to see in the cloud
custom_words = {"Amazon", "Flipkart", "Price", "Buy", "Product"} 
comment_words.update(custom_words)

# 4. Generate the Word Cloud
wordcloud = WordCloud(
    width = 800, 
    height = 800,
    background_color ='white',
    stopwords = comment_words,
    colormap = 'viridis', # Use 'viridis' for a professional tech look
    min_font_size = 10
).generate(text)

# 5. Create 'static' folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# 6. Save the image
wordcloud.to_file("static/product_wordcloud.png")

print("🎉 Success! Your word cloud is saved in the 'static' folder.")