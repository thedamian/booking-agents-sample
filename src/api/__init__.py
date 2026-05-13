"""
Booking Search API Package
==========================

AI-powered booking search backend for the DocumentDB workshop.

Modules:
- config: Environment configuration and settings
- database: DocumentDB connection and data access
- models: Pydantic models for requests/responses
- search: Vector and text search functionality
- chat: RAG-powered chat responses
- agents: Multi-agent system with LangGraph
- main: FastAPI application and endpoints

Usage:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

__all__ = [
    # Config
    "settings",
    # Database
    "db",
    "initialize_database",
    "get_database_status",
    # Models
    "Listing",
    "SearchRequest",
    "SearchResponse",
    "ChatRequest",
    "ChatResponse",
    # Search
    "search_listings",
    "get_search_capabilities",
    # Chat
    "generate_chat_response",
    # Agents
    "run_agent_query",
    "is_multi_agent_available",
]
