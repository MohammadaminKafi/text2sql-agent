
import logging as _logging

from contextlib import asynccontextmanager
import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv

from .config import settings
from .deps import init_components, shutdown_components
from .routers import report as report_router

from core.smartlog import init_logging, create_thread

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Load env from the nearest .env (project root in Docker Compose or local dev)
    load_dotenv(find_dotenv(usecwd=True), override=False)
    init_logging()  
    create_thread("system-init")  
    logging.getLogger(__name__).sysdebug("Service starting up")
    init_components()
    yield
    # Shutdown
    logging.getLogger(__name__).sysdebug("Service shutting down")
    shutdown_components()

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        lifespan=lifespan,   # use lifespan (no deprecated on_event)
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Per-request logging thread (optional but mirrors your "system-run") ---
    @app.middleware("http")
    async def request_log_thread(request: Request, call_next):
        req_id = request.headers.get("x-request-id") or str(uuid4())[:8]
        # Create a short-lived log thread per request (tweak naming if you prefer)
        create_thread(f"system-run-{req_id}")
        logging.getLogger(__name__).sysdebug(f"Handling {request.method} {request.url.path} [{req_id}]")
        resp = await call_next(request)
        return resp

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    app.include_router(report_router.router)
    return app

app = create_app()
