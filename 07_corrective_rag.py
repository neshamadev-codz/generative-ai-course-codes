from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Tokenization is the process of splitting text into smaller units called tokens.",
    "Vectorization converts text into numerical values.",
    "Embeddings represent meaning using dense numerical vectors.",
    "Refund Policy: Students can request a refund within 7 days.",
    "Assignments must be submitted before Sunday evening."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = model.encode(documents)

question = input("Ask your question: ")

def retrieve_document(query):
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, document_embeddings)[0]
    top_index = similarity_scores.argmax()
    return documents[top_index], similarity_scores[top_index]

retrieved_doc, score = retrieve_document(question)

print("\nFirst Retrieved Document:")
print(retrieved_doc)
print("Similarity Score:", score)

# Simple relevance check using important words
question_words = question.lower().split()

relevance_count = 0

for word in question_words:
    if word in retrieved_doc.lower():
        relevance_count += 1
        #relevance_count = relevance_count+1

if relevance_count == 0:
    print("\nRetrieved document seems weak.")
    print("Corrective RAG is rewriting the query and searching again...")

    rewritten_question = question + " course AI notes policy explanation"

    retrieved_doc, score = retrieve_document(rewritten_question)

    print("\nRewritten Query:")
    print(rewritten_question)

    print("\nNew Retrieved Document:")
    print(retrieved_doc)
    print("New Similarity Score:", score)
else:
    print("\nRetrieved document seems relevant.")

print("\nFinal Answer:")
print(retrieved_doc)
