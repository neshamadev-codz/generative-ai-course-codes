from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

context = """
You are an AI trainer.
Audience: Beginners.
Style: Simple English.
Use examples from Python programming.
"""

question = "Explain RAG."

response = client.responses.create(
    model="gpt-5.5",
    instructions=context,
    input=question
)

print(response.output_text)
