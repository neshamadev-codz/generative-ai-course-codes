from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

document_context = """
RAG stands for Retrieval-Augmented Generation.
It connects an LLM with external documents.
The system first retrieves relevant data, then the LLM generates an answer.
"""

question = "Explain RAG in simple words."

prompt = f"""
Use the following document context to answer.

Document:
{document_context}

Question:
{question}
"""

response = client.responses.create(
    model="gpt-5.5",
    input=prompt
)

print(response.output_text)
