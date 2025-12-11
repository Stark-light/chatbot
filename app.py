import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Document loader
from langchain_community.document_loaders import PyPDFLoader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vectorstore
from langchain_community.vectorstores import Chroma

# Retrieval
from langchain_classic.chains import RetrievalQA

# HuggingFace embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# OLLAMA (FREE)
from langchain_ollama import ChatOllama


# ---------------- FastAPI Setup ----------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None
qa_chain = None


# ------------- Request/Response Schemas -------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str


# ------------- Build PDF Retriever -----------------
def build_retriever_from_pdf(pdf_path: str):
    global vectorstore, qa_chain

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

    vs = Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        persist_directory="db"
    )

    # FAST / FREE / LOCAL MODEL
    llm = ChatOllama(
        model="llama3.2",  # or llama3.1, mistral, qwen, etc.
        temperature=0.1,
    )

    retriever = vs.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

    return vs, chain


# ------------- Upload PDF Endpoint -----------------

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, qa_chain

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    vectorstore, qa_chain = build_retriever_from_pdf(file_path)

    return {"status": "ok", "filename": file.filename}


# ------------- Chat Endpoint ------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global qa_chain
    if qa_chain is None:
        return ChatResponse(answer="Please upload a PDF first using /upload_pdf.")

    result = qa_chain({"query": req.message})
    return ChatResponse(answer=result["result"])


# ------------- Root Endpoint ------------------------

@app.get("/")
def root():
    return {"message": "FastAPI + LangChain + Ollama server is running (FREE & LOCAL)!"}
