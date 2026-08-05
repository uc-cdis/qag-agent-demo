from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Response
from orchestrator import RewooOrchestrator
from pydantic import BaseModel, Field
from pydantic_core import to_json


class InvokeRequest(BaseModel):
    query: str


def make_endpoint_app(*, orchestrator: RewooOrchestrator):
    app = FastAPI(title="QAG Rewoo Agent Endpoint API")

    @app.post("/invoke")
    async def invoke(req: InvokeRequest):
        try:
            result = await orchestrator.run_end_to_end(req.query, message_history=[])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Orchestrator failed: {e}")

        payload = {
            "response": result.answer,
            "run_id": result.run_id,
            "status": str(result.run_status),
            "execution_summary": getattr(result, "execution_summary", None),
            "final_evidence": getattr(result, "final_evidence", None),
            "token_usage": getattr(result, "token_usage", {}),
        }
        body = to_json(payload, fallback=str)
        return Response(content=body, media_type="application/json")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
