from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "RAG means Retrieval-Augmented Generation. It retrieves information before generating an answer.",
    "Vector databases store embeddings and help search by meaning.",
    "Tokenization splits text into smaller units called tokens.",
    "The AI course includes LLMs, RAG, embeddings, vector databases, and prompt engineering.",
    "Assignments must be submitted before Sunday evening."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = model.encode(documents)

chat_history = [
    {
        "role": "user",
        "content": "What is RAG?"
    },
    {
        "role": "assistant",
        "content": "RAG means Retrieval-Augmented Generation."
    }
]

current_question = input("Ask follow-up question: ")

combined_query = ""

for message in chat_history:
    combined_query += message["role"] + ": " + message["content"] + "\n"

combined_query += "user: " + current_question

# Use only the current question for retrieval.
# The full chat history biases the vector search toward previous topics
# (e.g. "RAG"), causing it to ignore what the current question is actually about.
query_embedding = model.encode([current_question])

similarity_scores = cosine_similarity(query_embedding, document_embeddings)[0]

top_index = similarity_scores.argmax()

print("\nCombined Query Used for Retrieval:")
print(combined_query)

print("\nRetrieved Document:")
print(documents[top_index])

print("\nSimple Answer:")
print(documents[top_index])
