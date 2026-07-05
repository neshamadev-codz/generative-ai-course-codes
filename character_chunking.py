text = """
Generative AI is transforming education.
Students use AI to learn programming.
AI also helps teachers create content.
Prompt engineering improves AI responses.
Vector databases store embeddings.
"""

chunk_size = 60

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}")
    print(chunk)

