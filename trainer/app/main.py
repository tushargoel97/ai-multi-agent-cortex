"""Host-side Cortex trainer API."""

from fastapi import FastAPI

from app.api.middleware.cors import configure_cors
from app.api.v1.router import router as api_v1_router

app = FastAPI(title="Cortex Trainer", version="0.1.0")
configure_cors(app)
app.include_router(api_v1_router, prefix="/api/v1")
