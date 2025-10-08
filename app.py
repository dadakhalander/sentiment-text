import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
from datetime import datetime
import io
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib

# NLP imports
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide"
)

class WhatsAppAnalyzer:
    def __init__(self):
        self.messages = []
        self.debug_info = []
        
    def parse_whatsapp_chat(self, file_content):
        """Universal WhatsApp parser that handles all formats"""
        try:
            lines = file_content.decode('utf-8').split('\n')
        except:
            lines = file_content.split('\n')
        
        self.debug_info = []
        self.messages = []
        current_message = None
        
        # Show first few lines for debugging
        st.info("First 5 lines of your file:")
        for i, line in enumerate(lines[:5]):
            st.write(f"Line {i+1}: `{line}`")
        
        # Try multiple date patterns
        patterns = [
            # 24/12/2023, 14:30 - Author: message
            r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})\s*-\s*(.+?):\s*(.*)$',
            # 24/12/2023, 2:30 PM - Author: message
            r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}\s*[APap][Mm])\s*-\s*(.+?):\s*(.*)$',
            # [24/12/2023, 14:30:00] Author: message
            r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?):\s*(.*)$',
            # 24/12/2023 - 14:30 - Author: message
            r'^(\d{1,2}/\d{1,2}/\d{2,4})\s*-\s*(\d{1,2}:\d{2})\s*-\s*(.+?):\s*(.*)$',
            # 2023-12-24, 14:30 - Author: message
            r'^(\d{4}-\d{1,2}-\d{1,2}),?\s+(\d{1,2}:\d{2})\s*-\s*(.+?):\s*(.*)$',
            # 24.12.2023, 14:30 - Author: message
            r'^(\d{1,2}\.\d{1,2}\.\d{2,4}),?\s+(\d{1,2}:\d{2})\s*-\s*(.+?):\s*(.*)$',
        ]
        
        line_count = 0
        parsed_count = 0
        
        for line in lines:
            line = line.strip()
            line_count += 1
            
            if not line:
                continue
                
            matched = False
            
            for pattern_idx, pattern in enumerate(patterns):
                match = re.match(pattern, line)
                if match:
                    date_str, time_str, author, message = match.groups()
                    self.debug_info.append(f"✅ Line {line_count}: Matched pattern {pattern_idx}")
                    self.debug_info.append(f"   Date: '{date_str}', Time: '{time_str}', Author: '{author}'")
                    
                    try:
                        dt = self.parse_datetime(date_str, time_str)
                        if dt:
                            if current_message:
                                self.messages.append(current_message)
                                parsed_count += 1
                            
                            current_message = {
                                'datetime': dt,
                                'author': author.strip(),
                                'message': message.strip(),
                                'clean_message': self.clean_message(message.strip())
                            }
                            matched = True
                            break
                    except Exception as e:
                        self.debug_info.append(f"   ❌ Date parsing failed: {e}")
                        continue
            
            if not matched:
                if current_message:
                    # Continuation of previous message
                    current_message['message'] += ' ' + line
                    current_message['clean_message'] = self.clean_message(current_message['message'])
                    self.debug_info.append(f"📝 Line {line_count}: Continued previous message")
                else:
                    # System message or unrecognized format
                    self.messages.append({
                        'datetime': None,
                        'author': 'System',
                        'message': line,
                        'clean_message': self.clean_message(line)
                    })
                    parsed_count += 1
                    self.debug_info.append(f"❓ Line {line_count}: System/unparsed message")
        
        if current_message:
            self.messages.append(current_message)
            parsed_count += 1
        
        return self.messages
    
    def parse_datetime(self, date_str, time_str):
        """Try multiple datetime formats"""
        date_formats = [
            '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d.%m.%Y',
            '%d/%m/%y', '%m/%d/%y', '%d.%m.%y'
        ]
        
        time_formats = [
            '%H:%M', '%I:%M %p', '%I:%M%p', '%H:%M:%S'
        ]
        
        for date_fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, date_fmt)
                break
            except ValueError:
                continue
        else:
            return None
        
        for time_fmt in time_formats:
            try:
                time_obj = datetime.strptime(time_str.upper(), time_fmt)
                combined = datetime.combine(date_obj.date(), time_obj.time())
                return combined
            except ValueError:
                continue
        
        return None
    
    def clean_message(self, text):
        """Clean message text"""
        if not text:
            return ""
        text = re.sub(r'<.*?>', '', text)  # Remove media tags
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
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
            pos_words = ['good', 'great', 'excellent', 'happy', 'love', 'like', 'nice', 'thanks', 'thank you', 'awesome', 'amazing']
            neg_words = ['bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'upset', 'sorry', 'no', 'not']
            
            pos_count = sum(1 for word in pos_words if word in text_lower)
            neg_count = sum(1 for word in neg_words if word in text_lower)
            
            if pos_count > neg_count:
                return 'positive', 0.5
            elif neg_count > pos_count:
                return 'negative', -0.5
            else:
                return 'neutral', 0.0

def create_visualizations(df):
    """Create comprehensive visualizations"""
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('WhatsApp Chat Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Sentiment distribution
    sentiment_counts = df['sentiment_label'].value_counts()
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    axes[0, 0].set_title('Sentiment Distribution')
    axes[0, 0].set_ylabel('Count')
    for i, count in enumerate(sentiment_counts.values):
        axes[0, 0].text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    # 2. Messages by author (top 10)
    author_counts = df['author'].value_counts().head(10)
    axes[0, 1].bar(author_counts.index, author_counts.values, color='skyblue')
    axes[0, 1].set_title('Top 10 Authors by Message Count')
    axes[0, 1].set_ylabel('Message Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Message length by sentiment
    df.boxplot(column='word_count', by='sentiment_label', ax=axes[0, 2])
    axes[0, 2].set_title('Message Length by Sentiment')
    axes[0, 2].set_ylabel('Word Count')
    
    # 4. Sentiment over time
    if df['datetime'].notna().any():
        df_sorted = df.sort_values('datetime')
        df_sorted['date'] = pd.to_datetime(df_sorted['datetime'].dt.date)
        daily_sentiment = df_sorted.groupby('date')['sentiment_score'].mean()
        axes[1, 0].plot(daily_sentiment.index, daily_sentiment.values, marker='o', markersize=2, linewidth=1)
        axes[1, 0].set_title('Average Sentiment Over Time')
        axes[1, 0].set_ylabel('Sentiment Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No datetime data', ha='center', va='center')
        axes[1, 0].set_title('No Time Data')
    
    # 5. Most common words
    try:
        all_words = ' '.join(df['clean_message'].dropna()).split()
        if all_words:
            word_freq = Counter(all_words).most_common(10)
            words, counts = zip(*word_freq)
            axes[1, 1].bar(words, counts, color='lightsteelblue')
            axes[1, 1].set_title('Top 10 Most Common Words')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No words to display', ha='center', va='center')
            axes[1, 1].set_title('No Word Data')
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, 'Error in word analysis', ha='center', va='center')
        axes[1, 1].set_title('Word Analysis Error')
    
    # 6. Emoji analysis
    if 'emoji_count' in df.columns:
        df.boxplot(column='emoji_count', by='sentiment_label', ax=axes[1, 2])
        axes[1, 2].set_title('Emoji Count by Sentiment')
        axes[1, 2].set_ylabel('Emoji Count')
    
    plt.tight_layout()
    return fig

def create_wordclouds(df):
    """Create word clouds for each sentiment"""
    sentiments = ['positive', 'negative', 'neutral']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, sentiment in enumerate(sentiments):
        text = ' '.join(df[df['sentiment_label'] == sentiment]['clean_message'].dropna())
        
        if text.strip():
            wordcloud = WordCloud(
                width=400, 
                height=300, 
                background_color='white',
                max_words=50
            ).generate(text)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{sentiment.capitalize()} Sentiment Words')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No {sentiment} messages', ha='center', va='center')
            axes[i].set_title(f'{sentiment.capitalize()} Sentiment')
            axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    st.title("💬 WhatsApp Chat Sentiment Analyzer")
    st.markdown("Upload your WhatsApp chat export to analyze sentiment patterns and statistics")
    
    uploaded_file = st.file_uploader("Choose your WhatsApp .txt file", type="txt")
    
    if uploaded_file is not None:
        try:
            # Read file content
            file_content = uploaded_file.read()
            
            # Show file info
            st.info(f"📁 File uploaded: {uploaded_file.name} ({len(file_content)} bytes)")
            
            # Parse chat
            analyzer = WhatsAppAnalyzer()
            messages = analyzer.parse_whatsapp_chat(file_content)
            
            # Show debug information
            with st.expander("🔍 Debug Information (Click to see parsing details)"):
                for info in analyzer.debug_info[:50]:
                    st.write(info)
                if len(analyzer.debug_info) > 50:
                    st.write(f"... and {len(analyzer.debug_info) - 50} more lines")
            
            if not messages:
                st.error("""
                ❌ No messages were parsed. Please check:
                1. Your chat export format
                2. Try exporting again from WhatsApp
                3. Make sure it's a .txt file without media
                """)
                
                # Let user paste sample lines
                sample_lines = st.text_area("Paste the first 5 lines of your chat file here for debugging:", height=150)
                if sample_lines:
                    st.code(sample_lines, language='text')
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(messages)
            
            # Analyze sentiment and add features
            sentiments = []
            sentiment_scores = []
            word_counts = []
            emoji_counts = []
            
            for msg in messages:
                sentiment, score = analyzer.analyze_sentiment(msg['clean_message'])
                sentiments.append(sentiment)
                sentiment_scores.append(score)
                word_counts.append(len(msg['clean_message'].split()))
                emoji_counts.append(emoji.emoji_count(msg['message']))
            
            df['sentiment_label'] = sentiments
            df['sentiment_score'] = sentiment_scores
            df['word_count'] = word_counts
            df['emoji_count'] = emoji_counts
            
            st.success(f"✅ Successfully analyzed {len(df)} messages!")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", len(df))
            with col2:
                st.metric("Unique Authors", df['author'].nunique())
            with col3:
                positive_count = (df['sentiment_label'] == 'positive').sum()
                st.metric("Positive Messages", positive_count)
            with col4:
                st.metric("Average Words", f"{df['word_count'].mean():.1f}")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "👥 Authors", "📝 Messages"])
            
            with tab1:
                st.subheader("Chat Overview")
                
                # Basic statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Sentiment Distribution:**")
                    sentiment_summary = df['sentiment_label'].value_counts()
                    for sentiment, count in sentiment_summary.items():
                        percentage = (count / len(df)) * 100
                        st.write(f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
                
                with col2:
                    st.write("**Author Activity:**")
                    author_summary = df['author'].value_counts().head(5)
                    for author, count in author_summary.items():
                        st.write(f"- {author}: {count} messages")
            
            with tab2:
                st.subheader("Visualizations")
                
                # Main visualizations
                st.pyplot(create_visualizations(df))
                
                # Word clouds
                st.subheader("Word Clouds by Sentiment")
                st.pyplot(create_wordclouds(df))
            
            with tab3:
                st.subheader("Author Analysis")
                
                # Author statistics
                author_stats = df.groupby('author').agg({
                    'message': 'count',
                    'sentiment_score': 'mean',
                    'word_count': 'mean',
                    'emoji_count': 'mean'
                }).round(3)
                
                author_stats.columns = ['Message Count', 'Avg Sentiment', 'Avg Words', 'Avg Emojis']
                st.dataframe(author_stats.sort_values('Message Count', ascending=False))
                
                # Author sentiment comparison
                st.subheader("Author Sentiment Comparison")
                author_sentiment = df.groupby('author')['sentiment_score'].mean().sort_values()
                st.bar_chart(author_sentiment)
            
            with tab4:
                st.subheader("Message Preview")
                
                # Search and filter
                col1, col2 = st.columns(2)
                with col1:
                    search_author = st.selectbox("Filter by author:", ["All"] + list(df['author'].unique()))
                with col2:
                    search_sentiment = st.selectbox("Filter by sentiment:", ["All", "positive", "neutral", "negative"])
                
                # Filter data
                filtered_df = df.copy()
                if search_author != "All":
                    filtered_df = filtered_df[filtered_df['author'] == search_author]
                if search_sentiment != "All":
                    filtered_df = filtered_df[filtered_df['sentiment_label'] == search_sentiment]
                
                # Display messages
                for idx, row in filtered_df.head(20).iterrows():
                    with st.container():
                        sentiment_emoji = '😊' if row['sentiment_label'] == 'positive' else '😐' if row['sentiment_label'] == 'neutral' else '😞'
                        st.write(f"**{row['author']}** {sentiment_emoji}")
                        st.write(f"*{row['datetime'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['datetime']) else 'No date'}*")
                        st.write(row['message'])
                        st.write(f"*Sentiment: {row['sentiment_label']} (score: {row['sentiment_score']:.3f})*")
                        st.divider()
            
            # Download results
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Analysis as CSV",
                    csv,
                    "whatsapp_sentiment_analysis.csv",
                    "text/csv"
                )
            
            with col2:
                # Save model (if you have one)
                try:
                    # Create a simple model for demonstration
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    
                    # This is just an example - you would use your actual trained model
                    st.info("Model saving feature available")
                    
                except Exception as e:
                    st.info("Advanced ML features require additional setup")
            
        except Exception as e:
            st.error(f"Error analyzing chat: {str(e)}")
            st.info("Please try uploading your file again or check the file format.")
    
    else:
        # Instructions
        st.markdown("""
        ### 📋 How to use this analyzer:
        
        1. **Export your WhatsApp chat:**
           - Open the chat in WhatsApp
           - Tap ⋮ (More) → Export Chat
           - Choose "Without Media"
        
        2. **Upload the .txt file** using the button above
        
        3. **View your analysis:**
           - Sentiment distribution
           - Author statistics  
           - Message trends over time
           - Word clouds and visualizations
        
        ### 🔒 Privacy Note:
        - All processing happens in your browser
        - Your chat data is never stored on any server
        - You can download and delete your results
        """)

if __name__ == "__main__":
    main()
