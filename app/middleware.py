"""
HTTP metrics middleware.

"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, REQUESTS_IN_PROGRESS


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records count, latency and in-flight requests for every call."""

    # Scraping /metrics would otherwise count itself, and Prometheus polls it
    # every ten seconds — enough to dominate the request count on a quiet
    # service and make the traffic panel meaningless.
    EXCLUDED = {"/metrics"}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Record every request except Prometheus scraping."""
        if request.url.path in self.EXCLUDED:
            return await call_next(request)

        started = time.perf_counter()
        status_code = 500

        REQUESTS_IN_PROGRESS.inc()

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        finally:
            endpoint = _route_template(
                request,
                request.url.path,
            )

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=str(status_code),
            ).inc()

            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(
                time.perf_counter() - started
            )

            REQUESTS_IN_PROGRESS.dec()


def _route_template(request: Request, fallback: str) -> str:
    """The matched route's path template, or the raw path if none matched."""
    route = request.scope.get("route")
    return getattr(route, "path", fallback) or fallback
