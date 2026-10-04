from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.llm import smart_route_request

router = APIRouter(tags=["Chat"])

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "auto" # 'auto' triggers our smart routing
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Standard OpenAI-compatible chat completion endpoint.
    If model='auto', we classify intent and route to the best model.
    """
    try:
        # Convert Pydantic model to dict for processing
        req_data = request.dict()
        
        # Route through our smart logic (Caching -> Classification -> LLM)
        response = await smart_route_request(req_data)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
