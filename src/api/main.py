# RAG System (Retrieval-Augmented Generation)

# Break down the structure into five steps:

from pathlib import Path
from google import genai
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Document loading
# Read raw files from disk and turn them into plain Python strings. This is the entry point — everything downstream depends on clean text here.

# Method 1
# practice_text = open("practice_text_rag.txt", "r")
# print(practice_text.read())
# practice_text.close()

# Method 2
practice_text = Path("practice_text_rag.txt").read_text()
print(practice_text)


# 2. Chunking
# Split the loaded text into smaller, manageable pieces for processing. LLMs have context limits, and small chunks make retrieval precise — you retrieve only the relevant paragraph, not the whole book.
def chunk_text(text: str, size: int = 500):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i : i + size])
    return chunks


chunks = chunk_text(practice_text)


# 3. Embedding
# Convert the text chunks into numerical vectors that can be used for similarity comparisons. (Semantic search) This is crucial for the retrieval step, as it allows you to find the most relevant chunks based on the user's query.

# Method 1
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(chunks)

# Method 2
# Load a small, fast model (free, runs locally)
model = SentenceTransformer("all-MiniLM-L6-v2")
# Embed all your chunks at once
embeddings = model.encode(chunks, show_progress_bar=True)
print(embeddings.shape)


# To see similarity between two chunks manually:
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


sim = cosine_sim(embeddings[0], embeddings[1])
print(f"Similarity: {sim: 3f}")


# 4. Vector Store & Retrieval
# Store the embedded vectors in a vector database and retrieve the most similar ones based on the user's query.
def retrieve(query: str, chunks: list, embeddings: np.ndarray, top_k: int = 3):
    # Embed the query using the same model
    query_embedding = model.encode([query])[0]

    # Score every chunk against the query
    scores = [cosine_sim(query_embedding, emb) for emb in embeddings]

    # Sort by score descending, take top_k
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices]


# 5. Generation - Question Answering
# Use the retrieved chunks to generate a response to the user's query. The LLM can now answer questions based on the context provided by the retrieved documents.
client = genai.Client()
