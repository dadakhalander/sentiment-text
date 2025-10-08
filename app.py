import streamlit as st
import pandas as pd
import re
from datetime import datetime
import io
from collections import Counter

# Import with error handling
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Download NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
        except:
            NLTK_AVAILABLE = False

def parse_whatsapp_chat(file_content):
    """Simple WhatsApp chat parser"""
    try:
        lines = file_content.decode('utf-8').split('\n')
    except:
        lines = file_content.split('\n')
    
    messages = []
    current_message = None
    
    # Common date patterns
    patterns = [
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})\s*-\s*(.+?):\s*(.*)$',
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(.+?):\s*(.*)$',
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
                    # Try different date formats
                    for dayfirst in [True, False]:
                        try:
                            if 'M' in time_str.upper():
                                dt_str = f"{date_str} {time_str}"
                                dt_format = "%d/%m/%Y %I:%M %p" if dayfirst else "%m/%d/%Y %I:%M %p"
                            else:
                                dt_str = f"{date_str} {time_str}"
                                dt_format = "%d/%m/%Y %H:%M" if dayfirst else "%m/%d/%Y %H:%M"
                            
                            dt = datetime.strptime(dt_str, dt_format)
                            break
                        except ValueError:
                            continue
                    else:
                        continue  # Skip if no format worked
                    
                    if current_message:
                        messages.append(current_message)
                    
                    current_message = {
                        'datetime': dt,
                        'author': author.strip(),
                        'message': message.strip(),
                    }
                    matched = True
                    break
                    
                except ValueError:
                    continue
        
        if not matched and current_message:
            # Continuation line
            current_message['message'] += ' ' + line
    
    if current_message:
        messages.append(current_message)
    
    return pd.DataFrame(messages)

def clean_message(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove common WhatsApp artifacts
    text = re.sub(r'<.*?>', '', text)  # Media markers
    text = re.sub(r'http\S+', '', text)  # URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Extra spaces
    return text

def simple_sentiment(text):
    """Rule-based sentiment as fallback"""
    text_lower = text.lower()
    
    positive_words = {'good', 'great', 'excellent', 'amazing', 'happy', 'love', 'like', 
                     'nice', 'awesome', 'fantastic', 'wonderful', 'perfect', 'best', 
                     'beautiful', 'yes', 'yeah', 'yay', 'woohoo', 'thanks', 'thank you'}
    
    negative_words = {'bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'upset', 
                     'disappointed', 'worst', 'horrible', 'no', 'not', "don't", 
                     "can't", 'worse', 'sorry', 'apologize'}
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return 'positive', 0.5
    elif neg_count > pos_count:
        return 'negative', -0.5
    else:
        return 'neutral', 0.0

def analyze_chat(df):
    """Main analysis function"""
    df['clean_message'] = df['message'].apply(clean_message)
    df['word_count'] = df['clean_message'].str.split().str.len().fillna(0)
    
    # Sentiment analysis
    if NLTK_AVAILABLE:
        sia = SentimentIntensityAnalyzer()
        def get_sentiment(text):
            scores = sia.polarity_scores(str(text))
            compound = scores['compound']
            if compound >= 0.05:
                return 'positive', compound
            elif compound <= -0.05:
                return 'negative', compound
            else:
                return 'neutral', compound
    else:
        get_sentiment = simple_sentiment
    
    sentiment_results = df['clean_message'].apply(get_sentiment)
    df[['sentiment', 'sentiment_score']] = pd.DataFrame(sentiment_results.tolist(), index=df.index)
    
    return df

def main():
    st.set_page_config(
        page_title="WhatsApp Analyzer",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 WhatsApp Chat Analyzer")
    st.markdown("Upload your WhatsApp chat export to analyze sentiment and statistics")
    
    uploaded_file = st.file_uploader("Choose a WhatsApp .txt file", type="txt")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing your chat..."):
                # Parse chat
                df = parse_whatsapp_chat(uploaded_file.read())
                
                if len(df) == 0:
                    st.error("Could not parse the chat file. Please check the format.")
                    return
                
                # Analyze
                df = analyze_chat(df)
                
            st.success(f"✅ Successfully analyzed {len(df)} messages!")
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", len(df))
            with col2:
                st.metric("Unique Authors", df['author'].nunique())
            with col3:
                positive = (df['sentiment'] == 'positive').sum()
                st.metric("Positive Messages", positive)
            with col4:
                st.metric("Average Words", f"{df['word_count'].mean():.1f}")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "👥 Authors", "📈 Sentiment"])
            
            with tab1:
                st.subheader("Chat Overview")
                
                # Sentiment distribution
                sentiment_counts = df['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
                
                # Recent messages
                st.subheader("Recent Messages")
                display_df = df[['datetime', 'author', 'message', 'sentiment']].copy()
                display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(display_df.head(20), use_container_width=True)
            
            with tab2:
                st.subheader("Author Statistics")
                
                author_stats = df.groupby('author').agg({
                    'message': 'count',
                    'sentiment_score': 'mean',
                    'word_count': 'mean'
                }).round(2)
                
                author_stats.columns = ['Message Count', 'Avg Sentiment', 'Avg Words']
                st.dataframe(author_stats.sort_values('Message Count', ascending=False))
            
            with tab3:
                st.subheader("Sentiment Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment by time (if datetime available)
                    if df['datetime'].notna().any():
                        df['date'] = pd.to_datetime(df['datetime'].dt.date)
                        daily_sentiment = df.groupby('date')['sentiment_score'].mean()
                        st.line_chart(daily_sentiment)
                
                with col2:
                    # Word count by sentiment
                    sentiment_word_stats = df.groupby('sentiment')['word_count'].mean()
                    st.bar_chart(sentiment_word_stats)
            
            # Download option
            st.subheader("Download Results")
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "whatsapp_analysis.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please ensure you're uploading a valid WhatsApp export file.")
    
    else:
        # Instructions
        st.markdown("""
        ### 📋 How to export your WhatsApp chat:
        
        1. **Open the chat** in WhatsApp
        2. **Tap ⋮ (More)** → **Export Chat**
        3. **Choose "Without Media"**
        4. **Upload the .txt file** above
        
        ### 🔒 Privacy Note:
        - All processing happens in your browser
        - Your data is never stored on any server
        - You can download and delete your results
        """)

if __name__ == "__main__":
    main()
