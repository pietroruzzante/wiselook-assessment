"""FastAPI application entrypoint: routes, static UI, exception handling."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.llm.client import LLMError
from app.observability.logging import configure_logging, request_logging_middleware

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

configure_logging()

app = FastAPI(title="Wiselook Assessment", version="0.1.0")

app.middleware("http")(request_logging_middleware)
app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})
