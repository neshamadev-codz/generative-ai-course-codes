from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

large_context = """
RAG means Retrieval-Augmented Generation.
It retrieves relevant documents from a database.
The documents are converted into embeddings.
Embeddings are stored in a vector database.
When a user asks a question, similar chunks are retrieved.
The LLM then uses those chunks to generate an answer.
"""

summary_response = client.responses.create(
    model="gpt-5.5",
    instructions="Compress the text into 3 short bullet points.",
    input=large_context
)

compressed_context = summary_response.output_text

question = "Explain RAG workflow."

final_prompt = f"""
Compressed Context:
{compressed_context}

Question:
{question}
"""

answer = client.responses.create(
    model="gpt-5.5",
    input=final_prompt
)

print(answer.output_text)
