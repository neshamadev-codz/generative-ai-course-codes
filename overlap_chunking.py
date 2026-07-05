# Overlap keeps some information from the previous chunk.
text = """
Generative AI is transforming education.
Students use AI to learn programming.
AI also helps teachers create content.
Prompt engineering improves AI responses.
Vector databases store embeddings.
Retrieval Augmented Generation improves answers.
"""

words = text.split()

chunk_size = 8
overlap = 3

chunks = []

start = 0

while start < len(words):

    end = start + chunk_size

    chunk = " ".join(words[start:end])

    chunks.append(chunk)

    start = end - overlap

print("Total Chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}")
    print(chunk)

