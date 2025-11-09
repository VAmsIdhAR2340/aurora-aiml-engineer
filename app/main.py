"""FastAPI application for Aurora QA System with async support"""
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.qa_engine import RAGQAEngine
from app.data_loader import AuroraDataLoader
from app.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
qa_engine = None
executor = ThreadPoolExecutor(max_workers=3)  # For CPU-bound operations

logging.getLogger("uvicorn.access").addFilter(
    lambda record: "/v1/models" not in record.getMessage()
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize QA engine on startup, cleanup on shutdown"""
    global qa_engine
    
    logger.info("Starting Aurora QA System...")
    
    try:
        # Validate configuration
        config.validate()
        
        # Initialize data loader
        data_loader = AuroraDataLoader(config.AURORA_API_URL)
        
        # Initialize QA engine (can take time, so run in executor)
        loop = asyncio.get_event_loop()
        qa_engine = await loop.run_in_executor(
            executor,
            lambda: RAGQAEngine(data_loader)
        )
        
        logger.info("Aurora QA System ready!")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    
    finally:
        logger.info("Shutting down Aurora QA System...")
        executor.shutdown(wait=True)


# Create FastAPI app
app = FastAPI(
    title="Aurora QA System",
    description="Semantic question-answering system for member messages using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language question about member data",
        examples=["When is Layla planning her trip to London?"]
    )


class AnswerResponse(BaseModel):
    """Response model for answers"""
    answer: str = Field(
        ...,
        description="Generated answer to the question"
    )


class DebugResponse(BaseModel):
    """Debug response with retrieved context"""
    answer: str
    retrieved_context: list


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    message: str


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint - API information
    
    Returns service metadata and available endpoints
    """
    return {
        "service": "Aurora QA System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "ask": "POST /ask - Ask questions about member data",
            "ask_debug": "POST /ask/debug - Ask questions with context (debug)",
            "health": "GET /health - Health check",
            "refresh": "POST /refresh - Refresh data from API",
            "docs": "GET /docs - Interactive API documentation",
        },
        "description": "Semantic Q&A system using RAG (Retrieval-Augmented Generation)"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns service health status
    """
    if qa_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA engine not initialized"
        )
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        message="Aurora QA System is operational"
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer natural language questions about member data
    
    This endpoint uses RAG (Retrieval-Augmented Generation) to:
    1. Find relevant member messages using semantic search
    2. Generate contextual answers using an LLM
    
    Example questions:
    - "When is Layla planning her trip to London?"
    - "How many cars does Vikram Desai have?"
    - "What are Amira's favorite restaurants?"
    
    Args:
        request: QuestionRequest containing the question
        
    Returns:
        AnswerResponse with the generated answer
        
    Raises:
        HTTPException: If QA engine not initialized or processing fails
    """
    if qa_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA engine not initialized"
        )
    
    try:
        # Run CPU/IO-bound operation in thread executor
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            executor,
            qa_engine.answer_question,
            request.question
        )
        
        return AnswerResponse(answer=answer)
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process question. Please try again."
        )


@app.post("/ask/debug", response_model=DebugResponse)
async def ask_question_debug(request: QuestionRequest):
    """
    Debug endpoint - returns answer with retrieved context
    
    Useful for development and understanding what messages were found
    relevant to the question. Shows the semantic search results.
    
    Args:
        request: QuestionRequest containing the question
        
    Returns:
        DebugResponse with answer and retrieved context
        
    Raises:
        HTTPException: If QA engine not initialized or processing fails
    """
    if qa_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA engine not initialized"
        )
    
    try:
        # Run operations in parallel using executor
        loop = asyncio.get_event_loop()
        
        # Execute both calls concurrently
        answer_task = loop.run_in_executor(
            executor,
            qa_engine.answer_question,
            request.question
        )
        context_task = loop.run_in_executor(
            executor,
            qa_engine.get_relevant_context,
            request.question,
            5
        )
        
        # Await both results
        answer, context = await asyncio.gather(answer_task, context_task)
        
        return DebugResponse(
            answer=answer,
            retrieved_context=context
        )
    
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/refresh")
async def refresh_data():
    """
    Refresh the QA engine with latest data from Aurora API
    
    Use this endpoint if new messages are added to the system.
    The vector store will be rebuilt with fresh data.
    
    Returns:
        Success message
        
    Raises:
        HTTPException: If QA engine not initialized or refresh fails
    """
    if qa_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA engine not initialized"
        )
    
    try:
        # Run refresh in executor (takes time)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            qa_engine.refresh_data
        )
        
        return {
            "message": "Data refreshed successfully",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh data"
        )


# Optional: Add metrics endpoint for monitoring
@app.get("/metrics")
async def get_metrics():
    """
    Get system metrics (optional, for monitoring)
    
    Returns basic statistics about the QA system
    """
    if qa_engine is None:
        return {"status": "not_initialized"}
    
    try:
        cache_status = qa_engine.data_loader.get_cache_status()
        return {
            "status": "operational",
            "messages_loaded": cache_status['message_count'],
            "cache_active": cache_status['cached'],
            "api_url": cache_status['api_url']
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/stats")
async def get_engine_stats():
    """Get QA engine statistics"""
    if qa_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA engine not initialized"
        )
    
    stats = qa_engine.get_stats()
    return stats

@app.get("/v1/models")
async def list_models():
    """Dummy endpoint to prevent 404 errors"""
    return {
        "object": "list",
        "data": [
            {
                "id": "sonar",
                "object": "model",
                "created": 1234567890,
                "owned_by": "perplexity"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
