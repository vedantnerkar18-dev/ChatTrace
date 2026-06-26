import pandas as pd
from collections import Counter
import re

def generate_stats(df):
    """
    Generate complete chat statistics from parsed dataframe
    """
    stats = {}

    # Basic counts
    stats['total_messages'] = len(df)
    stats['total_participants'] = df['sender'].nunique()
    stats['participants'] = df['sender'].unique().tolist()

    # Messages per person
    msg_counts = df['sender'].value_counts()
    stats['messages_per_person'] = msg_counts.to_dict()
    stats['most_active_person'] = msg_counts.index[0]
    stats['most_active_count'] = int(msg_counts.iloc[0])

    # Date analysis
    df['date_parsed'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df_valid = df.dropna(subset=['date_parsed'])

    if not df_valid.empty:
        # Busiest day
        day_counts = df_valid['date_parsed'].dt.date.value_counts()
        stats['busiest_day'] = str(day_counts.index[0])
        stats['busiest_day_count'] = int(day_counts.iloc[0])

        # Messages per weekday
        weekday_counts = df_valid['date_parsed'].dt.day_name().value_counts()
        stats['weekday_activity'] = weekday_counts.to_dict()

        # Chat duration
        stats['first_message_date'] = str(df_valid['date_parsed'].min().date())
        stats['last_message_date'] = str(df_valid['date_parsed'].max().date())
        duration = df_valid['date_parsed'].max() - df_valid['date_parsed'].min()
        stats['chat_duration_days'] = duration.days

        # Monthly activity
        df_valid['month_year'] = df_valid['date_parsed'].dt.strftime('%b %Y')
        monthly = df_valid['month_year'].value_counts().sort_index()
        stats['monthly_activity'] = monthly.to_dict()

    # Word analysis
    all_text = ' '.join(df['message'].astype(str).tolist()).lower()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'is', 'it', 'this', 'that', 'was', 'are',
        'be', 'have', 'had', 'do', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'we',
        'they', 'my', 'your', 'his', 'her', 'our', 'its', 'me', 'him',
        'us', 'them', 'so', 'if', 'as', 'by', 'from', 'up', 'about',
        'into', 'through', 'just', 'like', 'ok', 'okay', 'yes', 'no',
        'not', 'also', 'message', 'deleted', 'omitted', 'media', 'ha',
        'haha', 'hahaha', 'ah', 'oh', 'ye', 'yep', 'nah', 'na', 'bro',
        'hi', 'hey', 'hello', 'lol', 'omg', 'hmm', 'uh', 'um'
    }

    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
    filtered_words = [w for w in words if w not in stop_words]
    word_freq = Counter(filtered_words).most_common(10)
    stats['top_words'] = dict(word_freq)

    # Average message length
    df['msg_length'] = df['message'].astype(str).apply(len)
    stats['avg_message_length'] = round(df['msg_length'].mean(), 1)

    # Longest message
    longest_idx = df['msg_length'].idxmax()
    stats['longest_message'] = {
        'sender': df.loc[longest_idx, 'sender'],
        'message': df.loc[longest_idx, 'message'][:100] + '...',
        'length': int(df.loc[longest_idx, 'msg_length'])
    }

    return stats