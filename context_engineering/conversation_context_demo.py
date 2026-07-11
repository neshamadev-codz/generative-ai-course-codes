from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

conversation_history = """
User: What is Generative AI?
Assistant: Generative AI creates text, images, audio, video, or code.

User: What is an LLM?
Assistant: An LLM is a large language model trained to understand and generate text.
"""

question = "How is RAG connected to LLM?"

context = f"""
Previous Conversation:
{conversation_history}

Current Question:
{question}
"""

response = client.responses.create(
    model="gpt-5.5",
    instructions="Answer based on the previous conversation context.",
    input=context
)

print(response.output_text)

