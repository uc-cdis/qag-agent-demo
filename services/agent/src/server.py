import argparse
import os

import uvicorn
import yaml
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.applications import Starlette
from starlette.routing import Mount

from qag.v2._agent import make_agent_app as make_v2_app  # isort: skip
from qag.v3._agent import make_agent_app as make_v3_app  # isort: skip

PROVIDER_MAP: dict[
    str,
    tuple[
        type[AnthropicModel | GoogleModel | OpenAIChatModel],
        type[AnthropicProvider | GoogleProvider | OpenAIProvider],
    ],
] = {
    "anthropic": (AnthropicModel, AnthropicProvider),
    "google": (GoogleModel, GoogleProvider),
    "openai": (OpenAIChatModel, OpenAIProvider),
    "local": (
        OpenAIChatModel,
        OpenAIProvider,
    ),  # assume local is using OpenAI compatible API
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--root-path", default="/qag")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    models = dict()
    with open(args.model_config) as f:
        backends = yaml.safe_load(f)["backends"]
    for backend in backends:
        provider_type = backend["provider"]
        env_prefix = backend["env_prefix"]
        model_cls, provider_cls = PROVIDER_MAP[provider_type]
        api_key = os.environ.get(f"{env_prefix}_KEY", None)
        base_url = os.environ.get(f"{env_prefix}_URL", None)

        if provider_type == "local":
            assert (
                base_url is not None and base_url != ""
            ), f"Must set {env_prefix}_URL env var for {provider_type} backend"
        assert (
            api_key is not None and api_key != ""
        ), f"Must set {env_prefix}_KEY env var for {provider_type} backend"

        provider_inst = provider_cls(
            base_url=base_url,
            api_key=api_key,
        )
        models[backend["alias"]] = model_cls(
            model_name=backend["name"],
            provider=provider_inst,  # type: ignore
        )

    # because of dev UI's chat ID suffix logic, can't put v3 under a subpath of v2
    # e.g. if v2 gets /qag and v3 gets /qag/v3, the /v3 suffix will be interpreted
    # as a v2 chat ID even though v3 is still up and its own subpaths still work
    # get around this by just moving v2 to a subpath as well, need to update links
    v2_app = make_v2_app(
        models=models,
        mcp_url=args.mcp_url,
        retries=args.retries,
        root_path=args.root_path + "/v2",
    )
    v3_app = make_v3_app(
        models=models,
        mcp_url=args.mcp_url,
        retries=args.retries,
        root_path=args.root_path + "/v3",
    )
    inner = Starlette(
        routes=[
            Mount("/v2", app=v2_app),
            Mount("/v3", app=v3_app),
            # Mount("/sr-agent", app=sr_agent),
            # Mount("/sr-mcp", app=sr_mcp),
        ],
    )
    app = Starlette(routes=[Mount(args.root_path, app=inner)])
    uvicorn.run(app, host="0.0.0.0", port=args.port)
