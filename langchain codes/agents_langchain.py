import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Load API key from .env file (project root, two levels up from this script)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

model = ChatOpenAI(model="gpt-4.1")


# Tool 1
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


# Tool 2
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b


agent = create_agent(
    model=model,
    tools=[add_numbers, multiply_numbers],
    system_prompt="You are a helpful math assistant. Use tools when calculation is needed."
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Add 25 and 35, then multiply the result by 2."
        }
    ]
})

print(response["messages"][-1].content)
