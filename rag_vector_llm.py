from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import os

# -----------------------------
# 1. Load API Key
# -----------------------------
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 2. Documents
# -----------------------------
documents = [
    "Tokenization is the process of splitting text into smaller units called tokens.",
    "Vectorization converts text into numerical values so machine learning models can understand text.",
    "Embeddings are dense numerical vectors that represent the meaning of words or sentences.",
    "RAG means Retrieval-Augmented Generation. It retrieves relevant information before generating an answer.",
    "Prompt engineering is the process of writing clear instructions to get better AI responses."
]

# -----------------------------
# 3. Create Embedding Model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 4. Create Vector Database
# -----------------------------
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="rag_ai_notes")

# -----------------------------
# 5. Convert Documents into Embeddings
# -----------------------------
document_embeddings = embedding_model.encode(documents).tolist()

# -----------------------------
# 6. Store Documents in ChromaDB
# -----------------------------
collection.add(
    documents=documents,
    embeddings=document_embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# -----------------------------
# 7. User Question
# -----------------------------
question = input("Ask your question: ")

# -----------------------------
# 8. Convert Question into Embedding
# -----------------------------
question_embedding = embedding_model.encode([question]).tolist()

# -----------------------------
# 9. Retrieve Relevant Documents
# -----------------------------
results = collection.query(
    query_embeddings=question_embedding,
    n_results=2
)

retrieved_docs = results["documents"][0]

context = "\n".join(retrieved_docs)

print("\nRetrieved Context:")
print(context)

# -----------------------------
# 10. Send Context + Question to LLM
# -----------------------------
prompt = f"""
You are an AI tutor.

Answer the question using only the context below.
If the answer is not available in the context, say:
"I do not have enough information in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

response = openai_client.responses.create(
    model="gpt-5.5",
    input=prompt
)

print("\nFinal Answer:")
print(response.output_text)





