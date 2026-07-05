text = """
Generative AI is transforming education.
Students use AI to learn programming.
AI also helps teachers create content.
Prompt engineering improves AI responses.
Vector databases store embeddings.
"""

chunk_size = 8   # Number of words per chunk

words = text.split()

chunks = []

for i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i + chunk_size])
    chunks.append(chunk)

print("Total Chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}")
    print(chunk)
