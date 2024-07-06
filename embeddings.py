import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

# Example knowledge base
knowledge_base = [
    "Property values are influenced by various factors such as location, market trends, and property characteristics.",
    "You can predict property values by providing attributes like area, number of bedrooms, bathrooms, location, year built, and property type.",
    "To get an accurate property value estimate, consult with a real estate professional or use online tools."
]

# Generate embeddings for the knowledge base
embeddings = model.encode(knowledge_base)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def cosine_similarity(query_embedding, embeddings):
    query_embedding = np.array(query_embedding)
    embeddings = np.array(embeddings)
    dot_product = np.dot(embeddings, query_embedding)
    norm_query = np.linalg.norm(query_embedding)
    norm_embeddings = np.linalg.norm(embeddings, axis=1)
    similarities = dot_product / (norm_query * norm_embeddings)
    return similarities

# Function to find similar responses
def find_similar_response(query, embeddings, knowledge_base):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=1)
    return knowledge_base[I[0][0]]

# Function to add document to the index (in-memory)
def add_document_to_index(document):
    document_embedding = model.encode([document])[0]
    index.add(np.array([document_embedding]))
    knowledge_base.append(document)
