import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load McDonald's menu dataset (replace with actual file path)
df_mcdonalds = pd.read_csv('India_Menu.csv')  # Replace with actual file path

# Load Daily Food & Nutrition dataset (replace with actual file path)
df_daily_food = pd.read_csv('daily_food_nutrition_dataset.csv')  # Replace with actual file path

# Data Preprocessing - Extracting relevant columns
df_mcdonalds = df_mcdonalds[['Menu Items', 'Protein (g)', 'Total fat (g)']]
df_mcdonalds.columns = ['food_item', 'protein_mcd', 'fat_mcd']

df_daily_food = df_daily_food[['Food_Item', 'Protein (g)', 'Fat (g)', 'Category']]
df_daily_food.columns = ['food_item', 'protein_daily', 'fat_daily', 'category']

# Normalize the nutritional data for similarity comparison
scaler = StandardScaler()
df_mcdonalds[['protein_mcd', 'fat_mcd']] = scaler.fit_transform(df_mcdonalds[['protein_mcd', 'fat_mcd']])
df_daily_food[['protein_daily', 'fat_daily']] = scaler.fit_transform(df_daily_food[['protein_daily', 'fat_daily']])

# K-Nearest Neighbors for finding similar food items
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(df_daily_food[['protein_daily', 'fat_daily']])

# NLP: Use TF-IDF Vectorizer for food item name similarity
# Concatenating the food item columns from both datasets properly
combined_food_items = pd.concat([df_mcdonalds['food_item'], df_daily_food['food_item']], ignore_index=True)

tfidf = TfidfVectorizer(stop_words='english')
food_item_tfidf = tfidf.fit_transform(combined_food_items)

# Calculate cosine similarity between food item names
cosine_sim = cosine_similarity(food_item_tfidf)

# Deep Learning (BERT): Load a pre-trained BERT model for food description classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Example: binary classification for healthy vs unhealthy food

def bert_predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction

# Streamlit UI
st.title("Healthier Food Recommendation")

# Apply custom CSS for light blue background and aesthetic adjustments
st.markdown("""
    <style>
        .main {background-color: #f0f8ff;}  /* Light Blue background */
        .title {font-size: 28px; font-weight: bold; text-align: center;}
        .header {font-size: 22px; font-weight: bold; color: #333;}
        .sub-header {font-size: 18px; font-weight: normal; color: #555;}
        .result {font-size: 18px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# User selects a McDonald's food item
selected_item = st.selectbox("Select a McDonald's Food Item:", df_mcdonalds['food_item'].values)

# Get McDonald's nutritional info for the selected item
selected_mcd_item = df_mcdonalds[df_mcdonalds['food_item'] == selected_item]
selected_protein_mcd = selected_mcd_item['protein_mcd'].values[0]
selected_fat_mcd = selected_mcd_item['fat_mcd'].values[0]

# Find similar items in the Daily Food dataset using KNN
distances, indices = knn.kneighbors([[selected_protein_mcd, selected_fat_mcd]])

# Display McDonald's Item Info
st.write(f"**Selected McDonald's Food Item:** {selected_item}")
st.write(f"Protein: {selected_mcd_item['protein_mcd'].values[0]:.2f}g")
st.write(f"Fat: {selected_mcd_item['fat_mcd'].values[0]:.2f}g")

# Suggest Healthier Alternatives Based on Nutritional Info (KNN)
st.write("<h3 class='header'>HEALTHIER ALTERNATIVES:</h3>", unsafe_allow_html=True)

for i in range(len(indices[0])):
    recommended_item = df_daily_food.iloc[indices[0][i]]
    st.write(f"**Food Item:** {recommended_item['food_item']}")
    st.write(f"Category: {recommended_item['category']}")
    # Rescaling protein and fat back to original scale
    st.write(f"Protein: {recommended_item['protein_daily'] * scaler.scale_[0]:.2f}g")
    st.write(f"Fat: {recommended_item['fat_daily'] * scaler.scale_[1]:.2f}g")
    st.write("----")

# NLP-Based Matching: Find similar food items based on food item description
selected_item_index = df_mcdonalds[df_mcdonalds['food_item'] == selected_item].index[0]
similar_items = cosine_sim[selected_item_index]

# Get top 3 similar food items
top_3_indices = np.argsort(similar_items)[-3:][::-1]
st.write("<h3 class='header'>NLP-BASED SIMILAR FOOD ITEMS:</h3>", unsafe_allow_html=True)

for idx in top_3_indices:
    item_name = df_daily_food.iloc[idx]['food_item']
    st.write(f"**Similar Item:** {item_name}")

# Deep Learning (BERT) Model Prediction: Classify if McDonald's food is healthy or not
prediction = bert_predict(selected_item)
st.write("<h3 class='header'>PREDICTED HEALTHINESS:</h3>", unsafe_allow_html=True)
st.write(f"**{selected_item}:** {'Healthy' if prediction == 1 else 'Unhealthy'}")
