text = """
Generative AI is transforming education.
Students use AI to learn programming.
AI also helps teachers create content.
Prompt engineering improves AI responses.
Vector databases store embeddings.
"""

sentences = text.strip().split(".")

chunks = []

chunk = ""

for sentence in sentences:
    sentence = sentence.strip()

    if sentence:
        chunk += sentence + ". "

        if len(chunk.split(".")) >= 3:
            chunks.append(chunk)
            chunk = ""

if chunk:
    chunks.append(chunk)

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}")
    print(chunk)

