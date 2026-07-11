from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

memory = {
    "user_level": "beginner",
    "preferred_language": "simple English",
    "preferred_examples": "Python examples"
}

question = "Explain Vector Database."

context = f"""
User Memory:
Level: {memory["user_level"]}
Language Style: {memory["preferred_language"]}
Example Preference: {memory["preferred_examples"]}

Question:
{question}
"""

response = client.responses.create(
    model="gpt-5.5",
    instructions="Personalize the answer using the user memory.",
    input=context
)

print(response.output_text)
