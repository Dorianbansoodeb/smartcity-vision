"""FastAPI application factory.

Inference logic is not imported into route handlers except through ``cli.run``,
so the HTTP layer can be tested with a stubbed run function and the pipeline
can still be used without standing up a server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response

from smartcity_vision import __version__
from smartcity_vision.api.routes import router
from smartcity_vision.monitoring.metrics import record_request
from smartcity_vision.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging once when the API process starts."""
    setup_logging("INFO")
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application with routes and request metrics."""
    app = FastAPI(
        title="SmartCity Vision",
        version=__version__,
        description="Traffic-camera video analytics API.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def _count_requests(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        response = await call_next(request)
        record_request(request.method, request.url.path, response.status_code)
        return response

    return app


app = create_app()
