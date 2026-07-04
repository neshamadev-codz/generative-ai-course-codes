# 04_hybrid_rag.py


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


documents = [
    "Refund Policy: Students can request a refund within 7 days.",
    "Tokenization is the process of splitting text into smaller units called tokens.",
    "Vectorization converts text into numerical values.",
    "Assignments must be submitted before Sunday evening.",
    "A vector database stores embeddings and searches documents by meaning."
]


model = SentenceTransformer("all-MiniLM-L6-v2")


document_embeddings = model.encode(documents)


question = input("Ask your question: ")


STOP_WORDS = {"what", "is", "the", "a", "an", "of", "in", "to", "and", "or",
              "for", "on", "at", "by", "with", "from", "are", "was", "were",
              "be", "been", "it", "this", "that", "i", "you", "he", "she", "we"}


# Strip punctuation and remove stop words
question_words = [
    w.strip("?.,!;:'\"")
    for w in question.lower().split()
    if w.strip("?.,!;:'\"") not in STOP_WORDS
]


# -----------------------------
# 1. Keyword Score (normalized)
# -----------------------------
keyword_scores = []


for doc in documents:
    score = 0
    doc_lower = doc.lower()
    for word in question_words:
        if word in doc_lower:
            score += 1
    keyword_scores.append(score)


max_keyword = max(keyword_scores) if max(keyword_scores) > 0 else 1
norm_keyword_scores = [s / max_keyword for s in keyword_scores]


# -----------------------------
# 2. Vector Similarity Score
# -----------------------------
question_embedding = model.encode([question])


vector_scores = cosine_similarity(question_embedding, document_embeddings)[0]


# Normalize vector scores from [-1,1] to [0,1]
norm_vector_scores = [(s + 1) / 2 for s in vector_scores]


# -----------------------------
# 3. Combine Scores (weighted)
# -----------------------------
KEYWORD_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6


final_scores = [
    KEYWORD_WEIGHT * norm_keyword_scores[i] + VECTOR_WEIGHT * norm_vector_scores[i]
    for i in range(len(documents))
]


top_index = max(range(len(final_scores)), key=lambda i: final_scores[i])


print("\nQuestion:", question)


print("\nBest Retrieved Document:")
print(documents[top_index])


print("\nKeyword Score:", keyword_scores[top_index])
print("Vector Score:", vector_scores[top_index])
print("Final Hybrid Score:", final_scores[top_index])


print("\nSimple Answer:")
print(documents[top_index])



