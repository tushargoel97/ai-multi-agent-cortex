from fastapi import APIRouter

from cortex.api.v1.endpoints import assistants, runs, threads

router = APIRouter()
router.include_router(assistants.router)
router.include_router(threads.router)
router.include_router(runs.router)


@router.get("/health")
async def health():
    return {"ok": True}


@router.get("/info")
async def info():
    return {"version": "1", "flags": {"assistants": True, "crons": False}}
