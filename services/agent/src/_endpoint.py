from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from pydantic_core import to_json

from orchestrator import RewooOrchestrator  # isort: skip


class InvokeRequest(BaseModel):
    query: str
    backend: str


def make_endpoint_app(*, models, orchestrator: RewooOrchestrator):
    app = FastAPI(title="QAG Rewoo Agent Endpoint API")

    @app.post("/invoke")
    async def invoke(req: InvokeRequest):
        if not req.backend:
            raise HTTPException(
                400,
                f"Must specify 'backend' as one of {list(models.keys())}",
            )
        model = models.get(req.backend)

        try:
            result = await orchestrator.run_end_to_end(
                req.query, message_history=[], model=model
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Orchestrator failed: {e}")

        payload = {
            "response": result.answer,
            "run_id": result.run_id,
            "status": str(result.run_status),
            "token_usage": getattr(result, "token_usage", {}),
        }
        body = to_json(payload, fallback=str)
        return Response(content=body, media_type="application/json")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
