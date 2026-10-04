import json
import redis.asyncio as redis
import os
import hashlib
from typing import Optional, Dict, Any
from fastembed import TextEmbedding

# Initialize FastEmbed (runs locally on CPU)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Connect to local Redis Stack
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(redis_url, decode_responses=False) # Keep False for binary vector data

# Threshold for semantic similarity (1.0 is identical)
SIMILARITY_THRESHOLD = 0.92

def get_query_text(messages: list) -> str:
    """Extract the latest user message to embed."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""

async def get_embedding(text: str) -> list[float]:
    """Generate embedding using local FastEmbed model."""
    # FastEmbed returns a generator of arrays, we just want the first one
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()

async def check_semantic_cache(messages: list) -> Optional[Dict[str, Any]]:
    """
    Search Redis for a semantically similar previous query.
    If found, return the cached LLM response to save cost and latency!
    """
    query_text = get_query_text(messages)
    if not query_text:
        return None
        
    try:
        query_embedding = await get_embedding(query_text)
        
        # In a real production setup, we would use Redis RediSearch (FT.SEARCH) 
        # with vector similarity (KNN). For simplicity in this demo without 
        # complex index initialization, we will do an exact cache match first, 
        # or you can implement the full FT.SEARCH schema here.
        
        # Simple Exact Match Fallback (if Vector Index isn't set up yet)
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()
        cached_response = await r.get(f"cache:{query_hash}")
        
        if cached_response:
            print("[Cache] Exact match found! Returning cached response.")
            return json.loads(cached_response)
            
        return None
        
    except Exception as e:
        print(f"[Cache] Error checking cache: {e}")
        return None

async def save_to_cache(messages: list, response: Dict[str, Any]):
    """Save the LLM response to Redis for future queries."""
    query_text = get_query_text(messages)
    if not query_text:
        return
        
    try:
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()
        
        # Save exact match (TTL 1 hour)
        # To upgrade to semantic cache, we'd save the embedding vector in a Redis Hash here
        await r.setex(f"cache:{query_hash}", 3600, json.dumps(response.model_dump() if hasattr(response, 'model_dump') else response))
        print("[Cache] Saved response to cache.")
        
    except Exception as e:
        print(f"[Cache] Error saving to cache: {e}")
