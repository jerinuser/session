# query_index_chroma.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

HF_MODEL = "sergeyzh/BERTA"
GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash")  # or gemini-1.5-flash
PERSIST_DIR = "chroma_store"

texts = [
    "Hello, nice to meet you.",
    "LangChain simplifies the process of building applications with large language models",
    "Langchain Korean Tutorial...",
    "LangChainSimplifies the process...",
    "Retrieval-Augmented Generation (RAG) is an effective technique...",
]


# Same embedding model as build step
embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_MODEL,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

docs = [Document(page_content=t) for t in texts]

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
db.persist()
retriever = db.as_retriever(search_kwargs={"k": 3})


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True,
)

# Prompt must contain {context} and {input}
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the context to answer. "
               "If the answer is not in the context, say you don't know."),
    ("human", "Question: {input}\n\nContext:\n{context}")
])

# Turn retrieved docs into a single context string
format_docs = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))

# LCEL chain: {"context": retriever->format, "input": passthrough} | prompt | llm | str
chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("🔎 Ask questions (press Enter on empty line to exit).")
    while True:
        q = input("\nAsk: ").strip()
        if not q:
            break
        answer = chain.invoke(q)
        print("\nAnswer:", answer)

        # Show sources
        docs = retriever.invoke(q)
        print("\nSources:")
        for i, d in enumerate(docs, 1):
            snippet = d.page_content[:140].replace("\n", " ")
            src = d.metadata.get("source", "unknown")
            print(f"{i}. [{src}] {snippet}{'...' if len(d.page_content) > 140 else ''}")
