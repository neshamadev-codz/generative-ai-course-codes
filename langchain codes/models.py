import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API key from .env file (project root, two levels up from this script)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Create the model
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.7
)

# Send input to model
response = model.invoke("Explain Generative AI in simple words.")

print(response.content)
