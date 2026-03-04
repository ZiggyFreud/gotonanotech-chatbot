import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from rag import retrieve

app = FastAPI(title="GoToNanoTech Chatbot")

# CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(req: ChatRequest):
    question = req.message.strip()

    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    # Retrieve context from Chroma
    context, sources = retrieve(question, k=5)

    system_prompt = (
        "You are a helpful assistant for GoToNanoTech. "
        "Answer using the provided website context. "
        "If the answer is not in the context, say you do not know "
        "and suggest contacting GoToNanoTech."
    )

    user_prompt = f"""
Website context:
{context}

User question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources
    }