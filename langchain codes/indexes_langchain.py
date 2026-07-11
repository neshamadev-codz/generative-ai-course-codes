import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env file (project root, two levels up from this script)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Sample documents
docs = [
    Document(page_content="Generative AI creates new content such as text, images, code, audio, and video."),
    Document(page_content="RAG means Retrieval-Augmented Generation. It retrieves relevant data before answering."),
    Document(page_content="Vector databases store embeddings and help find similar content using similarity search."),
    Document(page_content="LangChain helps developers build LLM applications using models, prompts, chains, memory, and agents.")
]

# Convert documents into embeddings and store in FAISS vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

# User question
question = "What is RAG?"

# Retrieve relevant documents
retrieved_docs = vectorstore.similarity_search(question, k=2)

context = "\n".join([doc.page_content for doc in retrieved_docs])

# LLM
model = ChatOpenAI(model="gpt-4.1")

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
""")

rag_chain = prompt | model | StrOutputParser()

answer = rag_chain.invoke({
    "context": context,
    "question": question
})

print(answer)

