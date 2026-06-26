from sentence_transformers import SentenceTransformer
print("Downloading model during build...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded!")