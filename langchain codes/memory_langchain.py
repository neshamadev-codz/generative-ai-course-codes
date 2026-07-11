import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Load API key from .env file (project root, two levels up from this script)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

model = ChatOpenAI(model="gpt-4.1")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI trainer. Explain clearly."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model | StrOutputParser()

# In-memory store for chat sessions
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# First question
response1 = chain_with_memory.invoke(
    {"input": "What is RAG?"},
    config={"configurable": {"session_id": "student_1"}}
)

print("Response 1:")
print(response1)

# Follow-up question
response2 = chain_with_memory.invoke(
    {"input": "Explain it again with a real-time example."},
    config={"configurable": {"session_id": "student_1"}}
)

print("\nResponse 2:")
print(response2)

