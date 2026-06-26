from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("🧠 Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ AI model ready!")

def build_index(df):
    messages = df['message'].tolist()
    print(f"📊 Creating AI index for {len(messages)} messages...")
    embeddings = model.encode(messages, show_progress_bar=False, batch_size=16)
    print("✅ Index built!")
    return embeddings

def search(query, df, embeddings, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if similarities[idx] < 0.15:
            continue
        start = max(0, idx - 3)
        end = min(len(df), idx + 4)
        context = df.iloc[start:end].copy()
        context['is_match'] = False
        context.loc[idx, 'is_match'] = True
        results.append({
            'match_index': int(idx),
            'similarity_score': float(similarities[idx]),
            'matched_message': df.iloc[idx]['message'],
            'sender': df.iloc[idx]['sender'],
            'date': df.iloc[idx]['date'],
            'time': df.iloc[idx]['time'],
            'context': context.to_dict('records')
        })
    return results