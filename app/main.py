"""FastAPI application entrypoint: routes, static UI, exception handling."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.llm.client import LLMOutputError

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Wiselook Assessment", version="0.1.0")

app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(LLMOutputError)
async def llm_output_error_handler(request: Request, exc: LLMOutputError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})
