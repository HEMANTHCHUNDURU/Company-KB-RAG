from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend import initialize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG chain from backend
chain = initialize()

# ── Routes ────────────────────────────────────────────────
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(body: Question):
    answer = chain.invoke(body.question)
    return {"answer": answer}

@app.get("/")
async def root():
    return FileResponse("index.html")
