import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import agents, health, knowledge_bases
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.agent_runs import recover_interrupted_runs
from app.services.embeddings import warm_up
from app.services.ingestion import recover_interrupted_documents

# Show our own INFO logs (uvicorn only configures its own loggers)
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup (before `yield`) and once at shutdown (after)."""
    with SessionLocal() as db:
        recovered = recover_interrupted_documents(db)
        interrupted_runs = recover_interrupted_runs(db)
    if recovered:
        logger.warning(
            "%s unfinished document(s) from a previous run marked as error.", recovered
        )
    if interrupted_runs:
        logger.warning(
            "%s agent run(s) interrupted by the last stop marked as error.",
            interrupted_runs,
        )

    # Preload the default embedding model without delaying startup: the API
    # answers immediately, the model is ready a few seconds later.
    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(title="Agent Platform API", lifespan=lifespan)


# Middleware order matters: the LAST one added is the OUTERMOST. This catcher is
# added first, so it sits INSIDE the CORS middleware. Without it, an unhandled
# error produces a 500 that skips CORS: the browser hides it, and the frontend
# can only say "Can't reach the API" instead of "server error".
@app.middleware("http")
async def turn_crashes_into_json_500(request: Request, call_next) -> Response:
    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. The details are in the server logs."
            },
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(knowledge_bases.router)
app.include_router(agents.router)
