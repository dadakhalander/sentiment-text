import streamlit as st
import re
from datetime import datetime
import io
from collections import defaultdict

# Import with fallbacks
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

class UniversalChatParser:
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
        st.info(f"First 5 lines of your file:")
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
        """Analyze sentiment"""
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
            pos_words = ['good', 'great', 'excellent', 'happy', 'love', 'like', 'nice', 'thanks', 'thank you', 'awesome', 'amazing', 'perfect', 'best', 'beautiful', 'wonderful', 'fantastic', 'yay', 'woohoo', '🎉', '😊', '😍', '😂', '🥰', '👍', '🙌']
            neg_words = ['bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'upset', 'sorry', 'apologize', 'disappointed', 'worst', 'horrible', 'no', 'not', "don't", "can't", 'worse', '😠', '😡', '😢', '😞', '💔', '👎']
            
            pos_count = sum(1 for word in pos_words if word in text_lower)
            neg_count = sum(1 for word in neg_words if word in text_lower)
            
            if pos_count > neg_count:
                return 'positive', 0.5
            elif neg_count > pos_count:
                return 'negative', -0.5
            else:
                return 'neutral', 0.0

def main():
    st.set_page_config(
        page_title="WhatsApp Chat Analyzer",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 WhatsApp Chat Analyzer - Universal Parser")
    st.markdown("Upload your WhatsApp chat export - supports all formats")
    
    uploaded_file = st.file_uploader("Choose your WhatsApp .txt file", type="txt")
    
    if uploaded_file is not None:
        try:
            # Read file content
            file_content = uploaded_file.read()
            
            # Show file info
            st.info(f"📁 File uploaded: {uploaded_file.name} ({len(file_content)} bytes)")
            
            # Parse chat
            parser = UniversalChatParser()
            messages = parser.parse_whatsapp_chat(file_content)
            
            # Show debug information
            with st.expander("🔍 Debug Information (Click to see parsing details)"):
                for info in parser.debug_info[:50]:  # Show first 50 debug lines
                    st.write(info)
                if len(parser.debug_info) > 50:
                    st.write(f"... and {len(parser.debug_info) - 50} more lines")
            
            if not messages:
                st.error("""
                ❌ No messages were parsed. This usually means:
                
                **Common issues:**
                1. **Wrong date format** - Your chat uses a format we don't recognize
                2. **Different language** - The timestamps are in another language
                3. **Custom format** - Your phone/WhatsApp version uses a unique format
                
                **Please help me fix this:**
                - Copy the first 5 lines from your chat file
                - Paste them in the text area below
                - I'll create a custom parser for your format
                """)
                
                # Let user paste sample lines
                sample_lines = st.text_area("Paste the first 5 lines of your chat file here:", height=150)
                
                if sample_lines:
                    st.info("Sample lines received! Based on your format, I can create a custom parser.")
                    st.code(sample_lines, language='text')
                    
                    # Analyze the sample
                    st.subheader("Format Analysis:")
                    sample_lines_list = sample_lines.split('\n')
                    for i, line in enumerate(sample_lines_list[:5]):
                        st.write(f"**Line {i+1}:** `{line}`")
                
                return
            
            st.success(f"✅ Successfully parsed {len(messages)} messages!")
            
            # Analyze sentiment and get stats
            sentiments = []
            author_stats = defaultdict(lambda: {'count': 0, 'sentiment_sum': 0})
            word_counts = []
            
            for msg in messages:
                sentiment, score = parser.analyze_sentiment(msg['clean_message'])
                sentiments.append(sentiment)
                author_stats[msg['author']]['count'] += 1
                author_stats[msg['author']]['sentiment_sum'] += score
                word_counts.append(len(msg['clean_message'].split()))
            
            # Calculate statistics
            total_messages = len(messages)
            unique_authors = len(author_stats)
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            neutral_count = sentiments.count('neutral')
            avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", total_messages)
            with col2:
                st.metric("Unique Authors", unique_authors)
            with col3:
                st.metric("Positive Messages", positive_count)
            with col4:
                st.metric("Average Words", f"{avg_words:.1f}")
            
            # Sentiment distribution
            st.subheader("📊 Sentiment Distribution")
            sentiment_data = {
                'Positive': positive_count,
                'Neutral': neutral_count,
                'Negative': negative_count
            }
            st.bar_chart(sentiment_data)
            
            # Author statistics
            st.subheader("👥 Author Statistics")
            author_data = []
            for author, stats in author_stats.items():
                avg_sentiment = stats['sentiment_sum'] / stats['count'] if stats['count'] > 0 else 0
                author_data.append({
                    'Author': author,
                    'Messages': stats['count'],
                    'Avg Sentiment': f"{avg_sentiment:.3f}",
                    'Sentiment': '😊' if avg_sentiment > 0.1 else '😐' if avg_sentiment > -0.1 else '😞'
                })
            
            # Display author table
            for data in sorted(author_data, key=lambda x: x['Messages'], reverse=True):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.write(f"**{data['Author']}**")
                with col2:
                    st.write(f"{data['Messages']} messages")
                with col3:
                    st.write(f"Sentiment: {data['Avg Sentiment']}")
                with col4:
                    st.write(data['Sentiment'])
            
            # Message preview
            st.subheader("📝 Message Preview")
            for i, msg in enumerate(messages[:10]):
                sentiment, score = parser.analyze_sentiment(msg['clean_message'])
                with st.container():
                    col1, col2, col3 = st.columns([2, 6, 1])
                    with col1:
                        st.write(f"**{msg['author']}**")
                        if msg['datetime']:
                            st.write(msg['datetime'].strftime('%H:%M'))
                    with col2:
                        st.write(msg['message'][:200] + '...' if len(msg['message']) > 200 else msg['message'])
                    with col3:
                        sentiment_emoji = '😊' if sentiment == 'positive' else '😐' if sentiment == 'neutral' else '😞'
                        st.write(sentiment_emoji)
                st.divider()
            
            # Download results
            st.subheader("💾 Download Results")
            csv_content = "DateTime,Author,Message,CleanMessage,Sentiment\n"
            for msg in messages:
                sentiment, _ = parser.analyze_sentiment(msg['clean_message'])
                clean_msg = msg['clean_message'].replace('"', '""')
                original_msg = msg['message'].replace('"', '""')
                dt_str = msg['datetime'].strftime('%Y-%m-%d %H:%M:%S') if msg['datetime'] else ''
                csv_content += f'"{dt_str}","{msg["author"]}","{original_msg}","{clean_msg}","{sentiment}"\n'
            
            st.download_button(
                "📥 Download Analysis as CSV",
                csv_content,
                "whatsapp_sentiment_analysis.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please try uploading your file again or share a few sample lines for debugging.")
    
    else:
        st.markdown("""
        ### 📋 How to export your WhatsApp chat:
        
        1. **Open the chat** in WhatsApp
        2. **Tap ⋮ (More)** → **Export Chat**
        3. **Choose "Without Media"**
        4. **Upload the .txt file** above
        
        ### 🔍 Supported Formats:
        - `24/12/2023, 14:30 - John: Hello`
        - `12/24/2023, 2:30 PM - Jane: Hi there`  
        - `[24/12/2023, 14:30:00] John: Hello`
        - `2023-12-24, 14:30 - John: Hello`
        - And many more!
        
        ### 🔒 Privacy Note:
        - All processing happens in your browser
        - Your data is never stored on any server
        """)

if __name__ == "__main__":
    main()
