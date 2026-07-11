from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-5.5")

prompt = ChatPromptTemplate.from_template(
    "Create 5 bullet points about {topic} for a classroom training session."
)

parser = StrOutputParser()

# Chain using LCEL pipe operator
chain = prompt | model | parser

result = chain.invoke({
    "topic": "RAG and Vector Database"
})

print(result)
