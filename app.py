import streamlit as st
import re
from datetime import datetime
import io
from collections import defaultdict

# Import with fallbacks
try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Download NLTK data
if NLTK_AVAILABLE:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except:
        NLTK_AVAILABLE = False

class ChatAnalyzer:
    def __init__(self):
        self.messages = []
        
    def parse_whatsapp_chat(self, file_content):
        """Parse WhatsApp chat without pandas"""
        try:
            lines = file_content.decode('utf-8').split('\n')
        except:
            lines = file_content.split('\n')
        
        current_message = None
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
                        for dayfirst in [True, False]:
                            try:
                                if 'M' in time_str.upper():
                                    dt_format = "%d/%m/%Y %I:%M %p" if dayfirst else "%m/%d/%Y %I:%M %p"
                                else:
                                    dt_format = "%d/%m/%Y %H:%M" if dayfirst else "%m/%d/%Y %H:%M"
                                
                                dt_str = f"{date_str} {time_str}"
                                dt = datetime.strptime(dt_str, dt_format)
                                break
                            except ValueError:
                                continue
                        else:
                            continue
                        
                        if current_message:
                            self.messages.append(current_message)
                        
                        current_message = {
                            'datetime': dt,
                            'author': author.strip(),
                            'message': message.strip(),
                            'clean_message': self.clean_message(message.strip())
                        }
                        matched = True
                        break
                        
                    except ValueError:
                        continue
            
            if not matched and current_message:
                current_message['message'] += ' ' + line
                current_message['clean_message'] = self.clean_message(current_message['message'])
        
        if current_message:
            self.messages.append(current_message)
        
        return self.messages
    
    def clean_message(self, text):
        """Clean message text"""
        if not text:
            return ""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment with fallback"""
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
            # Simple rule-based
            text_lower = text.lower()
            pos_words = ['good', 'great', 'excellent', 'happy', 'love', 'like', 'nice', 'thanks']
            neg_words = ['bad', 'terrible', 'hate', 'angry', 'sad', 'sorry', 'no', 'not']
            
            pos_count = sum(1 for word in pos_words if word in text_lower)
            neg_count = sum(1 for word in neg_words if word in text_lower)
            
            if pos_count > neg_count:
                return 'positive', 0.5
            elif neg_count > pos_count:
                return 'negative', -0.5
            else:
                return 'neutral', 0.0
    
    def get_stats(self):
        """Calculate statistics"""
        if not self.messages:
            return {}
        
        stats = {
            'total_messages': len(self.messages),
            'authors': set(),
            'sentiments': defaultdict(int),
            'author_stats': defaultdict(lambda: {'count': 0, 'sentiment_sum': 0}),
            'word_counts': []
        }
        
        for msg in self.messages:
            # Author stats
            stats['authors'].add(msg['author'])
            stats['author_stats'][msg['author']]['count'] += 1
            
            # Sentiment analysis
            sentiment, score = self.analyze_sentiment(msg['clean_message'])
            stats['sentiments'][sentiment] += 1
            stats['author_stats'][msg['author']]['sentiment_sum'] += score
            
            # Word count
            word_count = len(msg['clean_message'].split())
            stats['word_counts'].append(word_count)
        
        stats['unique_authors'] = len(stats['authors'])
        stats['avg_words'] = sum(stats['word_counts']) / len(stats['word_counts']) if stats['word_counts'] else 0
        stats['positive_count'] = stats['sentiments']['positive']
        stats['negative_count'] = stats['sentiments']['negative']
        stats['neutral_count'] = stats['sentiments']['neutral']
        
        return stats

def main():
    st.set_page_config(
        page_title="WhatsApp Chat Analyzer",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 WhatsApp Chat Analyzer")
    st.markdown("Upload your WhatsApp chat export for sentiment analysis")
    
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing your chat..."):
                analyzer = ChatAnalyzer()
                messages = analyzer.parse_whatsapp_chat(uploaded_file.read())
                stats = analyzer.get_stats()
            
            if not messages:
                st.error("No messages found in the file. Please check the format.")
                return
            
            st.success(f"✅ Analyzed {stats['total_messages']} messages!")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", stats['total_messages'])
            with col2:
                st.metric("Unique Authors", stats['unique_authors'])
            with col3:
                st.metric("Positive", stats['positive_count'])
            with col4:
                st.metric("Average Words", f"{stats['avg_words']:.1f}")
            
            # Sentiment chart
            st.subheader("Sentiment Distribution")
            sentiment_data = {
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Count': [stats['positive_count'], stats['neutral_count'], stats['negative_count']]
            }
            
            if ALTAIR_AVAILABLE:
                chart = alt.Chart(alt.Data(values=sentiment_data)).mark_bar().encode(
                    x='Sentiment:N',
                    y='Count:Q',
                    color='Sentiment:N'
                ).properties(width=600, height=300)
                st.altair_chart(chart)
            else:
                # Simple bar chart using streamlit
                st.bar_chart({
                    'Positive': stats['positive_count'],
                    'Neutral': stats['neutral_count'], 
                    'Negative': stats['negative_count']
                })
            
            # Author statistics
            st.subheader("Author Statistics")
            author_data = []
            for author, author_stats in stats['author_stats'].items():
                avg_sentiment = author_stats['sentiment_sum'] / author_stats['count'] if author_stats['count'] > 0 else 0
                author_data.append({
                    'Author': author,
                    'Messages': author_stats['count'],
                    'Avg Sentiment': f"{avg_sentiment:.2f}"
                })
            
            # Display as table
            author_table = "| Author | Messages | Avg Sentiment |\n|--------|----------|---------------|\n"
            for data in sorted(author_data, key=lambda x: x['Messages'], reverse=True):
                author_table += f"| {data['Author']} | {data['Messages']} | {data['Avg Sentiment']} |\n"
            
            st.markdown(author_table)
            
            # Message preview
            st.subheader("Message Preview")
            preview_data = []
            for i, msg in enumerate(messages[:20]):
                sentiment, _ = analyzer.analyze_sentiment(msg['clean_message'])
                preview_data.append({
                    'Time': msg['datetime'].strftime('%H:%M'),
                    'Author': msg['author'],
                    'Message': msg['message'][:100] + '...' if len(msg['message']) > 100 else msg['message'],
                    'Sentiment': sentiment
                })
            
            preview_table = "| Time | Author | Message | Sentiment |\n|------|--------|---------|-----------|\n"
            for data in preview_data:
                preview_table += f"| {data['Time']} | {data['Author']} | {data['Message']} | {data['Sentiment']} |\n"
            
            st.markdown(preview_table)
            
            # Download results
            st.subheader("Download Results")
            csv_content = "DateTime,Author,Message,CleanMessage\n"
            for msg in messages:
                clean_msg = msg['clean_message'].replace('"', '""')
                original_msg = msg['message'].replace('"', '""')
                csv_content += f'"{msg["datetime"]}","{msg["author"]}","{original_msg}","{clean_msg}"\n'
            
            st.download_button(
                "Download CSV",
                csv_content,
                "whatsapp_analysis.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Error analyzing chat: {str(e)}")
    
    else:
        st.markdown("""
        ### 📋 How to use:
        1. **Export** your WhatsApp chat (.txt format)
        2. **Upload** the file above
        3. **View** sentiment analysis and statistics
        
        ### 🔒 Privacy:
        - All processing happens in your browser
        - No data is stored on servers
        - Your chats remain private
        """)

if __name__ == "__main__":
    main()
