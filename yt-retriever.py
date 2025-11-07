import os
from dotenv import load_dotenv

# --- LangChain / integrations ---
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ==== CONFIG ====
HF_MODEL = "sergeyzh/BERTA"  # your embedding model
GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash")
PERSIST_DIR = "chroma_youtube"  # folder to persist your vector store

# ---- utility: build retriever from a YouTube URL ----
def build_retriever_from_youtube(url: str, languages=("en", "en-US", "en-GB")):
    """
    Loads the YouTube transcript, splits it, embeds, and stores in Chroma.
    Returns a retriever you can use in a RAG chain.
    """
    # 1) Load transcript
    # add_video_info=True attaches title/channel/metadata
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
        language=list(languages),
    )
    docs = loader.load()  # one or more Documents

    if not docs:
        raise RuntimeError(
            "No transcript found. The video may have captions disabled "
            "or blocked for your locale."
        )

    # 2) Chunk into passages
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)

    # 3) Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model=HF_MODEL,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    # 4) Upsert into Chroma (namespace per-video so you can add many videos)
    #    We'll use the video ID as a collection-name suffix for neatness.
    #    If you want a single shared store across videos, keep one collection.
    collection_name = "yt_" + url.split("v=")[-1].split("&")[0] if "v=" in url else "yt_collection"

    db = Chroma(
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # Add docs (idempotent enough; duplicates are fine for a demo)
    db.add_documents(splits)
    db.persist()

    # 5) Return retriever
    return db.as_retriever(search_kwargs={"k": 5})

# ---- build the RAG chain (Gemini + retriever) ----
def build_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You answer questions strictly using the provided context. "
             "If the answer is not contained in the context, say you don't know."),
            ("human", "Question: {input}\n\nContext:\n{context}")
        ]
    )

    # Join retrieved docs for the prompt
    def join_docs(docs):
        # include timestamps if present
        lines = []
        for d in docs:
            ts = d.metadata.get("start_time")
            if ts is not None:
                lines.append(f"[t={int(ts)}s] {d.page_content}")
            else:
                lines.append(d.page_content)
        return "\n\n".join(lines)

    format_docs = RunnableLambda(join_docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    url = input("Paste YouTube URL: ").strip()
    if not url:
        print("No URL provided.")
        return

    try:
        retriever = build_retriever_from_youtube(url)
    except Exception as e:
        print(f"Failed to build index from YouTube: {e}")
        return

    chain = build_qa_chain(retriever)

    print("\n✅ Transcript indexed. Ask questions (empty line to quit).")
    while True:
        q = input("\nAsk: ").strip()
        if not q:
            break
        answer = chain.invoke(q)
        print("\nAnswer:", answer)

        # show sources/snippets so you can trust the answer
        docs = retriever.invoke(q)
        print("\nTop matches:")
        for i, d in enumerate(docs, 1):
            title = d.metadata.get("title", "video")
            start = d.metadata.get("start_time")
            stamp = f" @ {int(start)}s" if start is not None else ""
            snippet = d.page_content[:160].replace("\n", " ")
            print(f"{i}. {title}{stamp} — {snippet}{'...' if len(d.page_content)>160 else ''}")

if __name__ == "__main__":
    main()
