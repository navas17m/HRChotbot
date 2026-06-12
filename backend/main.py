"""
HR Policy Chatbot — FastAPI Backend
Endpoints:
  POST /chat          — Ask an HR question
  POST /ingest        — Re-ingest HR documents
  POST /clear/{sid}   — Clear session chat history
  GET  /health        — Health check
  GET  /sessions      — List active sessions
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat_history import ChatHistoryManager
from rag_pipeline import RAGPipeline

# ---- Logging Setup -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Global singletons (initialised at startup) --------------------------
rag: Optional[RAGPipeline]     = None
history: Optional[ChatHistoryManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global rag, history
    logger.info("🚀  Starting HR Policy Chatbot backend …")
    history = ChatHistoryManager()
    rag     = RAGPipeline()
    logger.info("✅  Backend ready.  Chunks in DB: %d", rag.get_collection_count())
    yield
    logger.info("🛑  Shutting down …")


# ---- FastAPI App ---------------------------------------------------------
app = FastAPI(
    title="HR Policy Chatbot API",
    description="RAG-powered chatbot for HR policy questions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Pydantic Models -----------------------------------------------------

class ChatRequest(BaseModel):
    query:      str              = Field(..., min_length=1, max_length=1000,
                                         example="How many vacation days do I get?")
    session_id: str              = Field(default_factory=lambda: str(uuid.uuid4()),
                                         example="user-abc-123")


class ChatResponse(BaseModel):
    answer:     str
    session_id: str
    sources:    List[str] = []
    turn_index: int       = 0


class IngestResponse(BaseModel):
    message:     str
    chunks_stored: int


class HealthResponse(BaseModel):
    status:         str
    chunks_in_db:   int
    active_sessions: int


# ---- Endpoints -----------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status          = "ok",
        chunks_in_db    = rag.get_collection_count() if rag else 0,
        active_sessions = len(history.list_sessions()) if history else 0,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    if not rag or not history:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialised yet. Please retry.",
        )

    try:
        session_history = history.get_history(req.session_id)
        result          = rag.chat(req.query, session_history)
        history.add_message(req.session_id, req.query, result["answer"])

        return ChatResponse(
            answer     = result["answer"],
            session_id = req.session_id,
            sources    = result.get("sources", []),
            turn_index = history.get_message_count(req.session_id),
        )
    except Exception as exc:
        logger.exception("Error processing chat request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(exc)}",
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest():
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not ready.")
    try:
        count = rag.ingest_documents()
        rag._build_chain()          # rebuild chain with fresh vectorstore
        return IngestResponse(
            message      = "Documents ingested and index rebuilt successfully.",
            chunks_stored= count,
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/clear/{session_id}", tags=["Chat"])
async def clear_session(session_id: str):
    if not history:
        raise HTTPException(status_code=503, detail="History manager not ready.")
    history.clear_history(session_id)
    return {"message": f"Chat history cleared for session '{session_id}'."}


@app.get("/sessions", tags=["Admin"])
async def list_sessions():
    if not history:
        return {"sessions": []}
    return {"sessions": history.list_sessions()}


# ---- Entry point ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
