# Program 1: Basic RAG + Vector DB without LLM API
# ChromaDB = Vector Database
# SentenceTransformer = Embedding Model
# Simple Python logic = Answer generator

# basic_rag_chromadb.py
# Simple understanding

import chromadb
from sentence_transformers import SentenceTransformer

# Below this similarity distance, a match is considered too weak to answer from.
DISTANCE_THRESHOLD = 1.0


# -------------------------------------------------------
# 1. Sample documents / knowledge base
# -------------------------------------------------------

documents = [
    "Tokenization is the process of splitting text into smaller units called tokens.",
    "Vectorization is the process of converting text into numerical values.",
    "Embeddings are numerical vectors that represent the meaning of text.",
    "RAG stands for Retrieval-Augmented Generation. It retrieves relevant information before generating an answer.",
    "A Vector Database stores embeddings and helps search information by meaning.",
    "LangChain is a framework used to build LLM applications.",
    "MCP connects AI applications with external tools, APIs, and data sources."
]

ids = [
    "doc1",
    "doc2",
    "doc3",
    "doc4",
    "doc5",
    "doc6",
    "doc7"
]


# -------------------------------------------------------
# 2. Load embedding model
# -------------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------------------------------
# 3. Create embeddings for documents
# -------------------------------------------------------

document_embeddings = embedding_model.encode(documents).tolist()


# -------------------------------------------------------
# 4. Create ChromaDB Vector Database
# -------------------------------------------------------

client = chromadb.PersistentClient(path="./basic_rag_vector_db")

# get_or_create avoids the delete/recreate dance and doesn't hide real errors
# behind a bare except.
collection = client.get_or_create_collection(name="ai_training_notes")


# -------------------------------------------------------
# 5. Store documents and embeddings in ChromaDB
# -------------------------------------------------------

# upsert (not add) so re-running the script doesn't raise a duplicate-ID error
# against documents already stored in the persistent collection.
collection.upsert(
    documents=documents,
    embeddings=document_embeddings,
    ids=ids
)

print("Documents stored successfully in ChromaDB.")


# -------------------------------------------------------
# 6. Ask user question
# -------------------------------------------------------

question = input("\nAsk your question: ").strip()

if not question:
    raise SystemExit("No question entered.")


# -------------------------------------------------------
# 7. Convert question into embedding
# -------------------------------------------------------

question_embedding = embedding_model.encode([question]).tolist()


# -------------------------------------------------------
# 8. Search relevant documents from Vector DB
# -------------------------------------------------------

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)

retrieved_docs = results["documents"][0]
retrieved_distances = results["distances"][0]


# -------------------------------------------------------
# 9. Simple RAG Answer Generation
# -------------------------------------------------------

print("\nRetrieved Documents:")

for i, (doc, distance) in enumerate(zip(retrieved_docs, retrieved_distances), start=1):
    print(f"{i}. ({distance:.4f}) {doc}")

print("\nFinal Answer:")

# Distance is checked because a nearest-neighbor search always returns
# something, even when nothing in the collection is actually relevant.
if retrieved_docs and retrieved_distances[0] <= DISTANCE_THRESHOLD:
    print("Based on the retrieved context:")
    print(retrieved_docs[0])
else:
    print("No relevant information found.")
