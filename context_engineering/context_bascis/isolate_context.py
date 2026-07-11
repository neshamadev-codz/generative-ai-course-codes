from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

system_rules = """
Explain simply.
Do not use advanced math.
"""

user_memory = """
User is learning Generative AI.
User knows basic Python.
"""

retrieved_docs = """
Context Engineering means giving the right information to the LLM.
It may include memory, tools, documents, and conversation history.
"""

question = "What is Context Engineering?"

prompt = f"""
[System Rules]
{system_rules}

[User Memory]
{user_memory}

[Retrieved Documents]
{retrieved_docs}

[Question]
{question}
"""

response = client.responses.create(
    model="gpt-5.5",
    input=prompt
)

print(response.output_text)
