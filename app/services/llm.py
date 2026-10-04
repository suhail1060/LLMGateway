import os
import litellm
from litellm import acompletion
from typing import Dict, Any

# Configure LiteLLM for telemetry and logging
litellm.success_callback = ["opentelemetry"]
litellm.failure_callback = ["opentelemetry"]

# Define our model tiers (using NVIDIA hosted models)
MODELS = {
    "cheap": "nvidia_nim/meta/llama3-8b-instruct",
    "powerful": "nvidia_nim/meta/llama3-70b-instruct"
}

def classify_intent(messages: list) -> str:
    """
    Very basic heuristic intent classification.
    In a real system, this could be a fast zero-shot classifier or local model.
    """
    # Just combining all user text for a simple check
    full_text = " ".join([m['content'].lower() for m in messages if m['role'] == 'user'])
    
    # Complex triggers
    complex_triggers = ["analyze", "code", "debug", "explain", "compare", "calculate"]
    if any(trigger in full_text for trigger in complex_triggers) or len(full_text) > 200:
        return "powerful"
    
    return "cheap"

async def smart_route_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core logic:
    1. Check Semantic Cache (TODO)
    2. Classify Intent
    3. Route via LiteLLM to the NVIDIA API
    4. Save to Cache (TODO)
    """
    
    messages = request_data.get("messages", [])
    requested_model = request_data.get("model", "auto")
    
    # 1. TODO: Check Semantic Cache here
    
    # 2. Determine target model
    if requested_model == "auto":
        tier = classify_intent(messages)
        target_model = MODELS[tier]
    else:
        target_model = requested_model

    print(f"[Router] Routing request to: {target_model}")

    # 3. Call Litellm (which calls NVIDIA NIM)
    try:
        # We use acompletion (async) for FastAPI
        response = await acompletion(
            model=target_model,
            messages=messages,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=request_data.get("temperature", 0.7),
            stream=request_data.get("stream", False)
        )
        
        # 4. TODO: Save to Semantic Cache here
        
        return response
    except Exception as e:
        print(f"[Router] Error calling LLM: {str(e)}")
        raise e
