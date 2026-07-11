from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-5.5")

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words for beginners with one example."
)

# Fill the prompt
formatted_prompt = prompt.invoke({
    "topic": "Vector Database"
})

# Send to model
response = model.invoke(formatted_prompt)

print(response.content)
