# 02_keyword_rag.py

documents = [
    "Refund Policy: Students can request a refund within 7 days.",
    "Assignment Policy: Assignments must be submitted before Sunday evening.",
    "Course Policy: Students must attend at least 80 percent of classes.",
    "Payment Policy: Course fees must be paid before batch confirmation.",
    "Certificate Policy: Certificates are issued after project completion."
]

question = input("Ask your question: ")
keyword = input("Enter keyword to search: ")

retrieved_docs = []

for doc in documents:
    if keyword.lower() in doc.lower():
        retrieved_docs.append(doc)

print("\nKeyword Used:", keyword)

print("\nRetrieved Documents:")

if retrieved_docs:
    for doc in retrieved_docs:
        print("-", doc)

    print("\nSimple Answer:")
    print(retrieved_docs[0])
else:
    print("No exact keyword match found.")
