"""
Booking Search API - Main Application
======================================

FastAPI backend for the AI-powered booking search workshop.
Demonstrates progressive functionality based on workshop completion:

- Pre-Module 0: Returns empty/demo data
- Post-Module 0: DocumentDB connection established
- Post-Module 1: Vector search available
- Post-Module 2: RAG chat responses
- Post-Module 3: Multi-agent system with LangGraph

Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .database import db, initialize_database, get_database_status
from .models import (
    Listing,
    SearchRequest,
    SearchResponse,
    SearchResult,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    CapabilityStatus,
)
from .search import search_listings, get_search_capabilities
from .chat import generate_chat_response, get_chat_history, clear_chat_history
from .agents import run_agent_query, is_multi_agent_available

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Application Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Booking Search API...")
    logger.info(f"Environment: {'Codespaces' if settings.is_codespaces else 'Local'}")

    # Initialize database connection
    await initialize_database()

    # Log capabilities
    db_status = get_database_status()
    search_caps = get_search_capabilities()

    logger.info(
        f"Database: {'Connected' if db_status['connected'] else 'Disconnected'}"
    )
    logger.info(
        f"Vector search: {'Available' if search_caps['vector_search'] else 'Unavailable'}"
    )
    logger.info(
        f"Text search: {'Available' if search_caps['text_search'] else 'Unavailable'}"
    )
    logger.info(
        f"Multi-agent: {'Available' if is_multi_agent_available() else 'Unavailable'}"
    )

    yield

    # Shutdown
    logger.info("Shutting down Booking Search API...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Booking Search API",
    description="AI-powered booking search with vector search and multi-agent capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns OK/Error status and detailed capability information.

    This is the first endpoint to test - if it works, the API is running.
    """
    db_status = get_database_status()
    search_caps = get_search_capabilities()

    # Determine overall status
    status = "ok" if db_status["connected"] else "degraded"

    return HealthResponse(
        status=status,
        message=(
            "API is running" if status == "ok" else "Running with limited functionality"
        ),
        capabilities=CapabilityStatus(
            database=db_status["connected"],
            vector_search=search_caps["vector_search"],
            text_search=search_caps["text_search"],
            static_data=search_caps["static_fallback"],
            chat=True,  # Always available (may use fallback)
            multi_agent=is_multi_agent_available(),
        ),
        database_info={
            "host": settings.DOCUMENTDB_HOST,
            "database": settings.DATABASE_NAME,
            "collection": settings.COLLECTION_NAME,
            "document_count": db_status.get("document_count", 0),
        },
    )


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Booking Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# Search Endpoints
# ============================================================================


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for listings using vector similarity or text search.

    Progressive behavior:
    - With vector index: Uses semantic search with embeddings
    - Without vector index: Falls back to text search
    - Without database: Falls back to static data

    Request body:
    - query: Search query string
    - limit: Maximum results (default: 10)
    - filters: Optional filters (property_type, bedrooms, max_price, amenities)
    """
    try:
        results = search_listings(
            query=request.query, limit=request.limit, filters=request.filters
        )

        search_caps = get_search_capabilities()

        return SearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            search_type=(
                "vector"
                if search_caps["vector_search"]
                else ("text" if search_caps["text_search"] else "static")
            ),
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse)
async def search_get(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
):
    """
    GET version of search endpoint for simple queries.
    """
    request = SearchRequest(query=query, limit=limit)
    return await search(request)


# ============================================================================
# Chat Endpoints
# ============================================================================


@app.post("/query_message", response_model=ChatResponse)
async def query_message(request: ChatRequest):
    """
    Process a chat message with RAG-powered responses.

    Progressive behavior:
    - With multi-agent (Module 3): Routes through specialist agents
    - With RAG (Module 2): Uses search results as context
    - Basic: Returns search results with simple formatting

    Request body:
    - message: User's question or request
    - session_id: Optional session ID for conversation tracking
    - use_agents: Whether to use multi-agent system (default: true if available)
    """
    try:
        session_id = request.session_id or "default"

        # Decide whether to use multi-agent system
        use_agents = request.use_agents and is_multi_agent_available()

        if use_agents:
            # Multi-agent path (Module 3)
            result = await run_agent_query(request.message, session_id)

            # Convert search results to SearchResult models
            search_results = []
            for r in result.get("search_results", []):
                if isinstance(r, dict):
                    listing_data = r.get("listing", r)
                    score = r.get("score", 1.0)
                    search_results.append(
                        SearchResult(listing=Listing(**listing_data), score=score)
                    )
                else:
                    search_results.append(r)

            return ChatResponse(
                message=result["response"],
                search_results=search_results,
                session_id=session_id,
                agent_path=result.get("agent_path", []),
                multi_agent=result.get("multi_agent", False),
            )
        else:
            # RAG path (Module 2)
            response = await generate_chat_response(request.message, session_id)

            # Also get search results for the frontend
            results = search_listings(request.message, limit=5)

            return ChatResponse(
                message=response,
                search_results=results,
                session_id=session_id,
                agent_path=[],
                multi_agent=False,
            )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
async def get_history(session_id: str = "default"):
    """Get chat history for a session."""
    history = get_chat_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.delete("/chat/history")
async def delete_history(session_id: str = "default"):
    """Clear chat history for a session."""
    clear_chat_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# ============================================================================
# Listings Endpoints
# ============================================================================


@app.get("/listings", response_model=List[Listing])
async def get_listings(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    category: Optional[str] = None,
    city: Optional[str] = None,
):
    """
    Get listings with optional filtering.

    This endpoint works even without search capabilities,
    returning data directly from the database or static files.
    """
    try:
        collection = db.get_collection()

        if collection is not None:
            # Build query
            query = {}
            if category:
                query["property_type"] = {"$regex": category, "$options": "i"}
            if city:
                query["neighborhood_overview"] = {"$regex": city, "$options": "i"}

            cursor = collection.find(query).skip(skip).limit(limit)

            from .models import normalize_listing

            listings = [normalize_listing(doc) for doc in cursor]
            return listings
        else:
            # Fallback to static data
            from .database import load_static_data

            data = load_static_data()

            # Apply filters
            if category:
                data = [
                    d
                    for d in data
                    if category.lower() in d.get("property_type", "").lower()
                ]
            if city:
                data = [
                    d
                    for d in data
                    if city.lower() in d.get("neighborhood_overview", "").lower()
                ]

            from .models import normalize_listing

            return [normalize_listing(d) for d in data[skip : skip + limit]]

    except Exception as e:
        logger.error(f"Listings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/listings/{listing_id}", response_model=Listing)
async def get_listing(listing_id: str):
    """Get a specific listing by ID."""
    try:
        collection = db.get_collection()

        if collection is not None:
            from bson import ObjectId

            try:
                doc = collection.find_one({"_id": ObjectId(listing_id)})
            except:
                doc = collection.find_one({"_id": listing_id})

            if doc:
                from .models import normalize_listing

                return normalize_listing(doc)
        else:
            # Fallback to static data
            from .database import load_static_data

            data = load_static_data()
            for item in data:
                if str(item.get("_id", item.get("id", ""))) == listing_id:
                    from .models import normalize_listing

                    return normalize_listing(item)

        raise HTTPException(status_code=404, detail="Listing not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Debug/Admin Endpoints (Development Only)
# ============================================================================

if settings.DEBUG:

    @app.get("/debug/config")
    async def debug_config():
        """Show current configuration (debug mode only)."""
        return {
            "host": settings.DOCUMENTDB_HOST,
            "port": settings.DOCUMENTDB_PORT,
            "database": settings.DATABASE_NAME,
            "collection": settings.COLLECTION_NAME,
            "is_codespaces": settings.is_codespaces,
            "cors_origins": settings.cors_origins[:5],  # First 5 only
            "openai_configured": bool(settings.OPENAI_API_KEY),
        }

    @app.get("/debug/static-data")
    async def debug_static_data():
        """Check static data availability (debug mode only)."""
        from .database import load_static_data

        data = load_static_data()
        return {
            "available": len(data) > 0,
            "count": len(data),
            "sample": data[0] if data else None,
        }


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred",
        },
    )


# ============================================================================
# Run directly (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
