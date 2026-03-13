from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# ── Step 1: Load Documents ────────────────────────────────
def load_documents(data_dir="../data"):
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    return documents

# ── Step 2: Chunk Documents ───────────────────────────────
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ── Step 3: Embed + Save to FAISS ─────────────────────────
def create_vectorstore(chunks, index_path="faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Saved to {index_path}/")
    return vectorstore

# ── Step 4: Load FAISS from disk ──────────────────────────
def load_vectorstore(index_path="faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded")
    return vectorstore

# ── Step 5: Build Retriever ───────────────────────────────
def build_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    print("✅ Retriever ready")
    return retriever

# ── Step 6: Format retrieved chunks ──────────────────────
def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

# ── Step 7: Build RAG Chain ───────────────────────────────
def build_chain(retriever):
    SYSTEM_PROMPT = """You are a helpful company knowledge assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
'I could not find this in our knowledge base.'
Always cite the source document and page number.

Context:
{context}"""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain ready")
    return chain

# ── Step 8: Ingest (run once) ─────────────────────────────
def ingest():
    documents = load_documents()
    chunks = chunk_documents(documents)
    create_vectorstore(chunks)

# ── Step 9: Initialize chain (called by app.py) ───────────
def initialize():
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    chain = build_chain(retriever)
    return chain
