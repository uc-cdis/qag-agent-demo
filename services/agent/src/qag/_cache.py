from cachetools import TTLCache
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route


def make_cache_app(cache: TTLCache) -> Starlette:
    async def get_cache_item(request: Request) -> Response:
        payload_id = request.path_params["payload_id"]
        try:
            payload = JSONResponse(cache[payload_id])
            cache[payload_id] = payload  # refresh
            return payload
        except KeyError:
            return JSONResponse({"detail": "Not found"}, status_code=404)

    return Starlette(
        routes=[Route("/{payload_id:str}", get_cache_item, methods=["GET"])]
    )
