import streamlit as st
import joblib
import re
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from newspaper import Article
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model
model = joblib.load('fake_news_detector.pkl')

# NewsAPI Key
NEWS_API_KEY = '5071a3a304914e4fbd4e18772b388b79'

# CSV file to save data
CSV_FILE = 'news_data.csv'

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

def fetch_live_news():
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        return articles
    else:
        st.error("Failed to fetch live news.")
        return []

def fetch_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def save_to_csv(text, prediction):
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['text', 'prediction'])

    new_row = pd.DataFrame({'text': [text], 'prediction': [prediction]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def load_from_csv():
    try:
        return pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=['text', 'prediction'])

def delete_from_csv(index):
    df = load_from_csv()
    df = df.drop(index)
    df.to_csv(CSV_FILE, index=False)

# Streamlit app
def main():
    st.title("Fake News Detector")
    st.write("Enter the news text below to check if it's real or fake.")

    news_text = st.text_area("Enter News Text Here", "")
    
    if st.button("Predict"):
        if news_text:
            clean_news = preprocess_text(news_text)
            prediction = model.predict([clean_news])[0]
            result = 'Fake News' if prediction == 1 else 'Real News'
            st.write(f"Prediction: **{result}**")
            save_to_csv(news_text, result)
        else:
            st.write("Please enter some text to predict.")

    st.write("## Live News")
    articles = fetch_live_news()
    for i, article in enumerate(articles):
        st.write(f"### {article['title']}")
        st.write(article['description'])
        if st.button(f"Predict for: {article['title']}", key=f'button_{i}'):
            clean_news = preprocess_text(article['description'] or article['title'])
            prediction = model.predict([clean_news])[0]
            result = 'Fake News' if prediction == 1 else 'Real News'
            st.write(f"Prediction: **{result}**")

    st.write("## Enter News URL")
    news_url = st.text_input("Enter the URL of the news article")

    if st.button("Fetch and Predict", key='fetch_predict'):
        if news_url:
            try:
                article_text = fetch_article_text(news_url)
                clean_article = preprocess_text(article_text)
                prediction = model.predict([clean_article])[0]
                result = 'Fake News' if prediction == 1 else 'Real News'
                st.write(f"Prediction: **{result}**")
                st.write(f"Article Text: {article_text}")
                save_to_csv(article_text, result)
            except Exception as e:
                st.error(f"Error fetching the article: {e}")
        else:
            st.write("Please enter a news URL to predict.")

    st.write("## Saved News Data")
    df = load_from_csv()
    if not df.empty:
        for i, row in df.iterrows():
            st.write(f"**News Text:** {row['text']}")
            st.write(f"**Prediction:** {row['prediction']}")
            if st.button(f"Delete Entry {i}", key=f'delete_button_{i}'):
                delete_from_csv(i)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
