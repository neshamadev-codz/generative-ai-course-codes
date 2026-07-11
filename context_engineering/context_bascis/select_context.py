from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

documents = [
    "RAG connects LLMs with external documents.",
    "Python is used for web development.",
    "Vector databases store embeddings for similarity search.",
    "HTML is used to create web pages."
]

question = "How does RAG use Vector DB?"

selected_context = []

for doc in documents:
    if "RAG" in doc or "Vector" in doc:
        selected_context.append(doc)

final_context = "\n".join(selected_context)

prompt = f"""
Relevant Context:
{final_context}

Question:
{question}
"""

response = client.responses.create(
    model="gpt-5.5",
    input=prompt
)

print(response.output_text)
print(prompt)