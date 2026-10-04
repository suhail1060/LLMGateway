import os
import litellm
from litellm import acompletion
from typing import Dict, Any
from app.services.cache import check_semantic_cache, save_to_cache

# Configure LiteLLM for telemetry and logging
litellm.success_callback = ["opentelemetry"]
litellm.failure_callback = ["opentelemetry"]

# Define our model tiers (using standard OpenAI format against NVIDIA base URL)
MODELS = {
    "cheap": "openai/google/gemma-4-31b-it",
    "powerful": "openai/google/gemma-4-31b-it" # Temporarily identical until more models unlock
}

def classify_intent(messages: list) -> str:
    """
    Very basic heuristic intent classification.
    """
    full_text = " ".join([m['content'].lower() for m in messages if m['role'] == 'user'])
    
    complex_triggers = ["analyze", "code", "debug", "explain", "compare", "calculate"]
    if any(trigger in full_text for trigger in complex_triggers) or len(full_text) > 200:
        return "powerful"
    
    return "cheap"

async def smart_route_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core logic:
    1. Check Semantic Cache
    2. Classify Intent
    3. Route via LiteLLM
    4. Save to Cache
    """
    
    messages = request_data.get("messages", [])
    requested_model = request_data.get("model", "auto")
    
    # 1. Check Cache First
    cached_response = await check_semantic_cache(messages)
    if cached_response:
        return cached_response
    
    # 2. Determine target model
    if requested_model == "auto":
        tier = classify_intent(messages)
        target_model = MODELS[tier]
    else:
        target_model = requested_model

    print(f"[Router] Cache miss. Routing request to: {target_model}")

    # 3. Call Litellm
    try:
        response = await acompletion(
            model=target_model,
            messages=messages,
            api_key=os.getenv("NVIDIA_API_KEY"),
            api_base="https://integrate.api.nvidia.com/v1",
            temperature=request_data.get("temperature", 0.7),
            stream=request_data.get("stream", False)
        )
        
        # 4. Save to Cache asynchronously (awaited here for simplicity)
        await save_to_cache(messages, response)
        
        return response
    except Exception as e:
        print(f"[Router] Error calling LLM: {str(e)}")
        raise e
