# 03_vector_rag.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Tokenization is the process of splitting text into smaller units called tokens.",
    "Vectorization converts text into numerical values so machines can understand text.",
    "Embeddings represent the meaning of words and sentences using dense vectors.",
    "RAG retrieves relevant information from documents before generating an answer.",
    "A vector database stores embeddings and helps search documents by meaning."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)

question = input("Ask your question: ")

question_embedding = model.encode([question])

similarity_scores = cosine_similarity(question_embedding, document_embeddings)[0]

top_index = similarity_scores.argmax()

print("\nQuestion:", question)
print("\nMost Relevant Document:")
print(documents[top_index])

print("\nSimilarity Score:")
print(similarity_scores[top_index])

print("\nSimple Answer:")
print(documents[top_index])
