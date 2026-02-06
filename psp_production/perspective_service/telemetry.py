"""OpenTelemetry tracing setup for Perspective Service."""

from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def telemetry_init(service_name, collector_endpoint=None, enable_tracing=True):
    """Initialize OpenTelemetry tracing."""
    from opentelemetry import trace

    if enable_tracing:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        resource = Resource.create(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        if collector_endpoint:
            exporter = OTLPSpanExporter(endpoint=collector_endpoint, insecure=True)
        else:
            exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    else:
        trace.set_tracer_provider(trace.NoOpTracerProvider())


def get_telemetry_context(headers):
    """Extract W3C traceparent context from HTTP headers."""
    try:
        return TraceContextTextMapPropagator().extract(
            {'traceparent': headers['traceparent']},
        )
    except (KeyError, NameError):
        return None
