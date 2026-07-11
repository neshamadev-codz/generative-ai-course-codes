from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

system_context = """
You are an AI trainer.
Explain concepts in simple beginner-friendly language.
Use real-time examples.
"""

user_question = "What is RAG?"

response = client.responses.create(
    model="gpt-5.5",
    instructions=system_context,
    input=user_question
)

print(response.output_text)

