from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def collect_context():
    return """
    Student Batch: Generative AI
    Topic Completed: RAG and Vector DB
    Next Topic: Agentic AI
    Assignment: Build a simple chatbot
    Deadline: Sunday
    """

def build_prompt(context, question):
    return f"""
    Context:
    {context}

    Question:
    {question}
    """

context = collect_context()

question = "Send a WhatsApp message to students about the assignment."

final_prompt = build_prompt(context, question)

response = client.responses.create(
    model="gpt-5.5",
    instructions="Write in a polite and professional tone.",
    input=final_prompt
)

print(response.output_text)

