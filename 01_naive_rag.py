# 01_naive_rag.py
# “Take user question → split into words → search documents for matching words 
# → show matching documents → return first match.”
documents = [
    "Tokenization is the process of splitting text into smaller units called tokens.",
    "Vectorization converts text into numerical values.",
    "Embeddings represent the meaning of text using numerical vectors.",
    "RAG retrieves relevant information before generating an answer.",
    "Prompt engineering means writing better instructions for AI models."
]

question = input("Ask your question: ")

question_words = question.lower().split()

retrieved_docs = []

for doc in documents:
    for word in question_words:
        if word in doc.lower():
            retrieved_docs.append(doc)
            break

print("\nRetrieved Documents:")

if retrieved_docs:
    for doc in retrieved_docs:
        print("-", doc)

    print("\nSimple Answer:")
    print(retrieved_docs[0])
else:
    print("No relevant document found.")
