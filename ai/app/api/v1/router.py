from fastapi import APIRouter

from app.api.v1.endpoints import admin, inference
from app.services import model_manager as mm

router = APIRouter()
router.include_router(admin.router)
router.include_router(inference.router)


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "cortex-ai",
        "loaded_model": mm.loaded_model(),
        "downloaded": [model["name"] for model in mm.list_downloaded()],
    }
