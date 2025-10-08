# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# NLP imports
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #f39c12; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💬 WhatsApp Chat Sentiment Analyzer</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("Upload your WhatsApp chat export file to analyze sentiment patterns.")

# File upload
uploaded_file = st.sidebar.file_uploader("📁 Upload WhatsApp Chat (.txt)", type="txt")

# WhatsApp chat parser (same as before)
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

# Text cleaning functions
def clean_whatsapp_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'image omitted|video omitted|audio omitted|sticker omitted', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    try:
        text = emoji.demojize(text, delimiters=(" ", " "))
    except:
        pass
    
    # Manual contraction expansion
    contractions_dict = {
        "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are",
        "'s": " is", "'d": " would", "'ll": " will", "'t": " not", "'ve": " have",
        "'m": " am"
    }
    for cont, expanded in contractions_dict.items():
        text = text.replace(cont, expanded)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment_scores(text):
        return sia.polarity_scores(str(text))
    
    df['clean_message'] = df['message'].apply(clean_whatsapp_text)
    df['sentiment_scores'] = df['clean_message'].apply(get_sentiment_scores)
    df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    
    def categorize_sentiment(compound_score):
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    df['sentiment_label'] = df['compound_score'].apply(categorize_sentiment)
    
    # Add features
    df['word_count'] = df['clean_message'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['clean_message'].apply(len)
    df['emoji_count'] = df['message'].apply(lambda x: emoji.emoji_count(str(x)) if pd.notna(x) else 0)
    
    return df

# Main app logic
if uploaded_file is not None:
    try:
        with st.spinner("🔄 Analyzing your WhatsApp chat..."):
            # Parse chat
            df = parse_whatsapp_chat(uploaded_file.read())
            
            if len(df) == 0:
                st.error("❌ Could not parse the chat file. Please check the format.")
                st.stop()
            
            # Analyze sentiment
            df = analyze_sentiment(df)
            
            # Store in session state
            st.session_state.df = df
            
        st.success(f"✅ Successfully analyzed {len(df)} messages!")
        
        # Display overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(df))
        
        with col2:
            st.metric("Unique Authors", df['author'].nunique())
        
        with col3:
            positive_count = (df['sentiment_label'] == 'positive').sum()
            st.metric("Positive Messages", positive_count)
        
        with col4:
            date_range = f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}"
            st.metric("Date Range", date_range)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", "😊 Sentiment Analysis", "👥 Author Analysis", 
            "📅 Timeline Analysis", "🔍 Word Analysis"
        ])
        
        with tab1:
            st.subheader("Chat Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Message count by author
                author_counts = df['author'].value_counts().head(10)
                fig = px.bar(author_counts, x=author_counts.index, y=author_counts.values,
                            title="Top 10 Authors by Message Count",
                            labels={'x': 'Author', 'y': 'Message Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment distribution
                sentiment_counts = df['sentiment_label'].value_counts()
                colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_sequence=colors)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Detailed Sentiment Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment by message length
                fig = px.box(df, x='sentiment_label', y='word_count',
                            title="Message Length by Sentiment",
                            labels={'sentiment_label': 'Sentiment', 'word_count': 'Word Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Emoji usage by sentiment
                emoji_by_sentiment = df.groupby('sentiment_label')['emoji_count'].mean()
                fig = px.bar(x=emoji_by_sentiment.index, y=emoji_by_sentiment.values,
                            title="Average Emoji Count by Sentiment",
                            labels={'x': 'Sentiment', 'y': 'Average Emoji Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment statistics
            st.subheader("Sentiment Statistics")
            sentiment_stats = df.groupby('sentiment_label').agg({
                'compound_score': ['mean', 'std', 'count'],
                'word_count': 'mean',
                'emoji_count': 'mean'
            }).round(3)
            st.dataframe(sentiment_stats)
        
        with tab3:
            st.subheader("Author-level Analysis")
            
            # Filter authors with minimum messages
            min_messages = st.slider("Minimum messages per author", 1, 50, 5)
            author_message_counts = df['author'].value_counts()
            valid_authors = author_message_counts[author_message_counts >= min_messages].index
            filtered_df = df[df['author'].isin(valid_authors)]
            
            if len(valid_authors) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Average sentiment by author
                    author_sentiment = filtered_df.groupby('author')['compound_score'].mean().sort_values()
                    fig = px.bar(x=author_sentiment.values, y=author_sentiment.index,
                                orientation='h', title="Average Sentiment by Author",
                                labels={'x': 'Average Sentiment Score', 'y': 'Author'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sentiment distribution by author
                    sentiment_by_author = pd.crosstab(filtered_df['author'], filtered_df['sentiment_label'], normalize='index')
                    fig = px.bar(sentiment_by_author, x=sentiment_by_author.index, y=sentiment_by_author.columns,
                                title="Sentiment Distribution by Author",
                                labels={'x': 'Author', 'y': 'Proportion', 'variable': 'Sentiment'})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No authors meet the minimum message threshold.")
        
        with tab4:
            st.subheader("Timeline Analysis")
            
            if df['datetime'].notna().any():
                # Daily sentiment trend
                df['date'] = pd.to_datetime(df['date'])
                daily_sentiment = df.groupby('date')['compound_score'].mean().reset_index()
                
                fig = px.line(daily_sentiment, x='date', y='compound_score',
                            title="Daily Average Sentiment Over Time",
                            labels={'date': 'Date', 'compound_score': 'Sentiment Score'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekly pattern
                df['day_of_week'] = df['datetime'].dt.day_name()
                df['hour'] = df['datetime'].dt.hour
                
                col1, col2 = st.columns(2)
                
                with col1:
                    day_sentiment = df.groupby('day_of_week')['compound_score'].mean().reindex([
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ])
                    fig = px.bar(x=day_sentiment.index, y=day_sentiment.values,
                                title="Average Sentiment by Day of Week",
                                labels={'x': 'Day of Week', 'y': 'Average Sentiment'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    hour_sentiment = df.groupby('hour')['compound_score'].mean()
                    fig = px.line(x=hour_sentiment.index, y=hour_sentiment.values,
                                title="Average Sentiment by Hour of Day",
                                labels={'x': 'Hour of Day', 'y': 'Average Sentiment'})
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("Word Analysis")
            
            # Word clouds for each sentiment
            sentiments = ['positive', 'negative', 'neutral']
            col1, col2, col3 = st.columns(3)
            
            for i, sentiment in enumerate(sentiments):
                with [col1, col2, col3][i]:
                    st.subheader(f"{sentiment.capitalize()} Messages")
                    sentiment_text = ' '.join(df[df['sentiment_label'] == sentiment]['clean_message'].dropna())
                    
                    if sentiment_text.strip():
                        wordcloud = WordCloud(width=300, height=200, background_color='white', max_words=50).generate(sentiment_text)
                        
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.info(f"No {sentiment} messages to analyze")
            
            # Most frequent words
            st.subheader("Most Frequent Words")
            all_text = ' '.join(df['clean_message'].dropna())
            if all_text.strip():
                words = all_text.split()
                word_freq = Counter(words).most_common(20)
                
                words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                fig = px.bar(words_df.head(10), x='Frequency', y='Word', orientation='h',
                            title="Top 10 Most Frequent Words")
                st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Download Results")
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Analysis Results (CSV)",
            data=csv,
            file_name="whatsapp_sentiment_analysis.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please make sure you're uploading a valid WhatsApp chat export file.")

else:
    # Welcome screen
    st.markdown("""
    ## 📱 How to Use This Analyzer
    
    1. **Export your WhatsApp chat**: 
       - Open the chat in WhatsApp
       - Tap ⋮ (More) → Export Chat
       - Choose "Without Media"
    
    2. **Upload the .txt file** using the sidebar
    
    3. **Explore the analysis** through different tabs
    
    ### 📊 What You'll Discover:
    
    - **Overall sentiment** of your conversations
    - **Most active participants**
    - **Sentiment trends** over time
    - **Word patterns** in positive/negative/neutral messages
    - **Author-specific** sentiment analysis
    
    ### 🔒 Privacy Note:
    - All processing happens in your browser
    - Your chat data is not stored on any server
    - You can download the results and delete the analysis
    """)
    
    # Sample data option
    if st.button("🎲 Try with Sample Data"):
        sample_data = """24/12/2023, 14:30 - Alice: Hey everyone! Christmas party tonight? 🎄🎉
24/12/2023, 14:31 - Bob: Yes! So excited! 😊
24/12/2023, 14:32 - Charlie: I can't make it, sorry 😔
24/12/2023, 14:33 - Alice: Oh no! That's too bad Charlie
24/12/2023, 14:34 - Bob: We'll miss you! But more cake for us! 🍰
24/12/2023, 14:35 - David: What time does it start?
24/12/2023, 14:36 - Alice: 7 PM at my place! Bring your favorite drinks! 🥂
24/12/2023, 14:37 - Eve: This is going to be amazing! So happy! 🌟"""
        
        # Create sample file
        sample_file = io.BytesIO(sample_data.encode())
        sample_file.name = "sample_chat.txt"
        
        # Rerun with sample file
        st.session_state.sample_file = sample_file
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ using Streamlit | Your chats remain private and secure"
    "</div>",
    unsafe_allow_html=True
)
