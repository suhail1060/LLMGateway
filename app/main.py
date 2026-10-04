from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from app.routers import chat
from app.core.telemetry import setup_telemetry
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenTelemetry
setup_telemetry()

app = FastAPI(
    title="LLM Gateway",
    description="Smart Router with Caching & Observability",
    version="1.0.0"
)

# Include our API routes
app.include_router(chat.router, prefix="/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Instrument the FastAPI app for OpenTelemetry tracing
FastAPIInstrumentor.instrument_app(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
