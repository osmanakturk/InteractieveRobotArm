# ai_server/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="AI Server", version="1.0.0")


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "service": "ai_server", "message": "alive"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
