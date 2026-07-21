import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from app.api.middleware.cors import configure_cors
from app.api.v1.router import router as api_v1_router
from app.config import settings
from app.services import model_manager as mm

logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.default_local_model:
        try:
            await mm.load_model(settings.default_local_model)
        except Exception as error:
            logger.warning("Could not preload default model: %s", error)

    async def idle_watchdog():
        while True:
            await asyncio.sleep(60)
            try:
                await mm.maybe_unload_idle()
            except Exception:
                logger.exception("Idle unload check failed")

    watchdog = asyncio.create_task(idle_watchdog(), name="idle-unload")
    try:
        yield
    finally:
        watchdog.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog


app = FastAPI(title="Cortex Local LLM", version="0.1.0", lifespan=lifespan)
configure_cors(app)
app.include_router(api_v1_router, prefix="/api/v1")
