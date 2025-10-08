# robust_app.py
import streamlit as st
import pandas as pd
import re
from datetime import datetime
import io
from collections import Counter

# Import with fallbacks
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    st.warning("Emoji package not available. Emoji analysis disabled.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not available. Some charts disabled.")

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Interactive charts disabled.")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available. Using simple sentiment analysis.")

# Parser function (same as before)
def parse_whatsapp_chat(file_content, dayfirst=True):
    lines = file_content.decode('utf-8').split('\n')
    messages = []
    current_message = None
    
    patterns = [
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})\s*-\s*(.+?):\s*(.*)$',
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(.+?):\s*(.*)$',
        r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?):\s*(.*)$',
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        matched = False
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                date_str, time_str, author, message = match.groups()
                
                try:
                    if 'M' in time_str.upper():
                        dt_str = f"{date_str} {time_str}"
                        dt_format = "%d/%m/%Y %I:%M %p" if dayfirst else "%m/%d/%Y %I:%M %p"
                    else:
                        dt_str = f"{date_str} {time_str}"
                        dt_format = "%d/%m/%Y %H:%M" if dayfirst else "%m/%d/%Y %H:%M"
                    
                    dt = datetime.strptime(dt_str, dt_format)
                    
                    if current_message:
                        messages.append(current_message)
                    
                    current_message = {
                        'datetime': dt,
                        'date': dt.date(),
                        'time': dt.time(),
                        'author': author.strip(),
                        'message': message.strip(),
                        'is_system': False,
                        'is_media': False
                    }
                    matched = True
                    break
                    
                except ValueError:
                    continue
        
        if not matched:
            if current_message:
                if line.startswith('<') and line.endswith('>') or 'media omitted' in line.lower():
                    current_message['is_media'] = True
                current_message['message'] += ' ' + line
            else:
                messages.append({
                    'datetime': None,
                    'date': None,
                    'time': None,
                    'author': 'System',
                    'message': line,
                    'is_system': True,
                    'is_media': False
                })
    
    if current_message:
        messages.append(current_message)
    
    return pd.DataFrame(messages)

# Simple text cleaning
def clean_whatsapp_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'image omitted|video omitted|audio omitted|sticker omitted', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Sentiment analysis with fallback
def analyze_sentiment(text):
    if NLTK_AVAILABLE:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound
    else:
        # Simple rule-based fallback
        text_lower = text.lower()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'happy', 'love', 'like', 'nice']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'upset']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive', 0.5
        elif negative_count > positive_count:
            return 'negative', -0.5
        else:
            return 'neutral', 0.0

# Main app
def main():
    st.set_page_config(page_title="WhatsApp Analyzer", layout="wide")
    st.title("💬 WhatsApp Chat Analyzer")
    
    uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type="txt")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing your chat..."):
                df = parse_whatsapp_chat(uploaded_file.read())
                df['clean_message'] = df['message'].apply(clean_whatsapp_text)
                
                # Analyze sentiment
                sentiment_results = df['clean_message'].apply(analyze_sentiment)
                df[['sentiment_label', 'sentiment_score']] = pd.DataFrame(sentiment_results.tolist(), index=df.index)
                
                # Basic features
                df['word_count'] = df['clean_message'].apply(lambda x: len(str(x).split()))
                df['message_length'] = df['clean_message'].apply(len)
            
            st.success(f"✅ Analyzed {len(df)} messages!")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", len(df))
            with col2:
                st.metric("Unique Authors", df['author'].nunique())
            with col3:
                st.metric("Positive", (df['sentiment_label'] == 'positive').sum())
            with col4:
                st.metric("Negative", (df['sentiment_label'] == 'negative').sum())
            
            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment_label'].value_counts()
            
            if PLOTLY_AVAILABLE:
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                            title="Message Sentiment Distribution")
                st.plotly_chart(fig)
            else:
                st.bar_chart(sentiment_counts)
            
            # Author analysis
            st.subheader("Author Statistics")
            author_stats = df.groupby('author').agg({
                'message': 'count',
                'sentiment_score': 'mean',
                'word_count': 'mean'
            }).round(2)
            author_stats.columns = ['Message Count', 'Avg Sentiment', 'Avg Words']
            st.dataframe(author_stats.sort_values('Message Count', ascending=False))
            
            # Message preview
            st.subheader("Message Preview")
            st.dataframe(df[['datetime', 'author', 'message', 'sentiment_label']].head(20))
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Analysis Results (CSV)",
                data=csv,
                file_name="whatsapp_analysis.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error analyzing chat: {str(e)}")
            st.info("Please make sure you're uploading a valid WhatsApp export file.")

if __name__ == "__main__":
    main()
