import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.rag import ask_question
from src.search import ingest_documents
from src.schemas import QuestionRequest, QuestionResponse


# Configure application logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting document knowledge base synchronisation.")

    try:
        ingest_documents()
        logger.info("Document knowledge base synchronisation completed.")
    except Exception:
        logger.exception("Knowledge base synchronisation failed.")
        raise

    yield

    logger.info("Shutting down Semantic Document Search API.")


app = FastAPI(
    title="Semantic Document Search API",
    description=(
        "A RAG-powered document question-answering API "
        "using ChromaDB, Sentence Transformers and Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    logger.info("Received question: %s", request.question)

    try:
        result = ask_question(request.question)

        logger.info(
            "Question processed successfully. References retrieved: %d",
            len(result["references"])
        )

        return result

    except Exception as error:
        logger.exception("Failed to process question.")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        ) from error
