# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import pandas as pd

# # Load AI model once when app starts
# print("🧠 Loading AI model...")
# model = SentenceTransformer('all-MiniLM-L6-v2')
# print("✅ AI model ready!")


# def build_index(df):
#     """
#     Takes parsed chat dataframe and creates
#     AI embeddings (numeric meaning) for every message
#     """
#     messages = df['message'].tolist()
    
#     print(f"📊 Creating AI index for {len(messages)} messages...")
#     embeddings = model.encode(messages, show_progress_bar=True)
#     print("✅ Index built!")
    
#     return embeddings


# def search(query, df, embeddings, top_k=5):
#     """
#     Takes a search query and finds most relevant
#     messages using AI semantic understanding
#     """
#     # Convert search query to AI embedding
#     query_embedding = model.encode([query])
    
#     # Calculate similarity between query and all messages
#     similarities = cosine_similarity(query_embedding, embeddings)[0]
    
#     # Get top matching indices
#     top_indices = np.argsort(similarities)[::-1][:top_k]
    
#     results = []
    
#     for idx in top_indices:
#         # Skip results below minimum relevance threshold
#         if similarities[idx] < 0.15:
#             continue

#         # Get surrounding context (3 messages before and after)
#         start = max(0, idx - 3)
#         end = min(len(df), idx + 4)
        
#         context = df.iloc[start:end].copy()
#         context['is_match'] = False
#         context.loc[idx, 'is_match'] = True
        
#         results.append({
#             'match_index': int(idx),
#             'similarity_score': float(similarities[idx]),
#             'matched_message': df.iloc[idx]['message'],
#             'sender': df.iloc[idx]['sender'],
#             'date': df.iloc[idx]['date'],
#             'time': df.iloc[idx]['time'],
#             'context': context.to_dict('records')
#         })
    
#     return results


# def test_searcher():
#     """Test the AI search with sample data"""
#     from parser import parse_whatsapp_chat
    
#     # Use the test file created by parser
#     df = parse_whatsapp_chat('test_chat.txt')
    
#     # Build AI index
#     embeddings = build_index(df)
    
#     # Test search
#     print("\n🔍 Searching for: 'cooking ingredients'")
#     print("-" * 40)
    
#     results = search("cooking ingredients", df, embeddings)
    
#     for i, result in enumerate(results):
#         print(f"\nResult {i+1} (Score: {result['similarity_score']:.2f})")
#         print(f"📅 {result['date']} {result['time']}")
#         print(f"👤 {result['sender']}: {result['matched_message']}")
#         print(f"\n--- Context ---")
#         for msg in result['context']:
#             marker = "👉" if msg['is_match'] else "  "
#             print(f"{marker} {msg['sender']}: {msg['message']}")


# if __name__ == "__main__":
#     test_searcher()


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("🧠 Loading AI model...")

from transformers import AutoTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Download ONNX model
onnx_path = hf_hub_download(
    repo_id=MODEL_NAME,
    filename="onnx/model.onnx"
)
session = ort.InferenceSession(onnx_path)
print("✅ AI model ready!")


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask[..., np.newaxis].astype(float)
    return (token_embeddings * mask).sum(1) / mask.sum(1)


def encode(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        outputs = session.run(None, dict(encoded))
        embeddings = mean_pooling(outputs[0], encoded['attention_mask'])
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)


def build_index(df):
    messages = df['message'].tolist()
    print(f"📊 Creating AI index for {len(messages)} messages...")
    embeddings = encode(messages)
    print("✅ Index built!")
    return embeddings


def search(query, df, embeddings, top_k=5):
    query_embedding = encode([query])
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