import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

def setup_telemetry():
    """
    Configure OpenTelemetry to send traces to our local Jaeger instance.
    """
    # Set the service name for Jaeger
    resource = Resource.create(attributes={
        SERVICE_NAME: "llm-gateway"
    })
    
    # Initialize the TracerProvider
    provider = TracerProvider(resource=resource)
    
    # Configure the OTLP exporter (Jaeger)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    
    # Add the batch processor to the provider
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    # Set the global tracer provider
    trace.set_tracer_provider(provider)
    
    print(f"[Telemetry] OpenTelemetry configured. Exporting to {otlp_endpoint}")
