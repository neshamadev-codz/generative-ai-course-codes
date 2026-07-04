from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Vectorization converts text into numerical values.",
    "Embeddings represent the meaning of text using dense vectors.",
    "Tokenization splits text into smaller units called tokens.",
    "Machine learning models need numerical input.",
    "RAG retrieves relevant information before generating an answer."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = model.encode(documents)

question = input("Ask your question: ")

queries = [
    question,
    "What is vectorization?",
    "How is text converted into numbers?",
    "How do machine learning models understand words?"
]

retrieved_docs = set()

for query in queries:
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, document_embeddings)[0]

    top_index = similarity_scores.argmax()
    retrieved_docs.add(documents[top_index])

print("\nOriginal Question:")
print(question)

print("\nGenerated Query Versions:")

for query in queries:
    print("-", query)

print("\nRetrieved Documents:")

for doc in retrieved_docs:
    print("-", doc)

print("\nSimple Answer:")
for doc in retrieved_docs:
    print(doc)
