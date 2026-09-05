from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import settings_store
from .api.routes import router as api_router
from .db.session import create_tables

app = FastAPI(title="多Agent中医辨证论治系统", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup() -> None:
    settings_store.load_runtime()
    try:
        await create_tables()
    except Exception:
        pass


app.include_router(api_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "llm_configured": str(settings_store.public_settings()["configured"]).lower()}
