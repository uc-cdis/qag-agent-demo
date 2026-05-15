from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_core import to_json


class InvokeRequest(BaseModel):
    query: str
    backend: str


class InvokeResponse(BaseModel):
    response: str
    backed: str
    steps: list[dict]


def extract_steps(messages) -> list[dict]:
    steps = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    steps.append(
                        {
                            "type": "tool_call",
                            "tool": part.tool_name,
                            "args": part.args,
                            "id": part.tool_call_id,
                        }
                    )
                elif isinstance(part, TextPart):
                    steps.append(
                        {
                            "type": "model_text",
                            "content": part.content,
                        }
                    )
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    steps.append(
                        {
                            "type": "tool_return",
                            "tool": part.tool_name,
                            "id": part.tool_call_id,
                            "content": part.content,
                        }
                    )
    return steps


def make_endpoint_app(*, models, agent):
    app = FastAPI(title="QAG Agent Endpoint API")

    @app.post("/invoke")
    async def invoke(req: InvokeRequest):
        if not req.backend:
            raise HTTPException(
                400,
                f"Multiple backends configured {list(models)}; specify 'backend'",
            )
        model = models.get(req.backend)
        if model is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown backend: {req.backend}. Available: {list(models.keys())}",
            )
        result = await agent.run(req.query, model=model)
        payload = {
            "response": result.output,
            "backend": model,
            "steps": extract_steps(result.new_messages()),
        }
        body = to_json(payload, fallback=str)
        return Response(content=body, media_type="application/json")

    @app.get("/health")
    async def health():
        return {"status": "ok", "backends": list(models.keys())}

    return app
