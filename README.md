# QAG Agent Demo

At its core, this demo relies on a strong, general-purpose, reasoning LLM and an MCP server with specialized tools. The LLM must be capable of doing multistep tool calling; such capabilities are simplified by using an LLM agent that is stateful. We demonstrate the use of the revamped MCP server using 2 LLMs: locally served `gpt-oss-120b` or Claude Sonnet.

## Table of Contents
1. [Conceptual Reframing](#conceptual-reframing)
1. [Revamped MCP Architecture](#revamped-mcp-architecture)
    1. [Guiding Use Cases](#guiding-use-cases)
    1. [Modularization and Reusable Components](#modularization-and-reusable-components)
    1. [MCP Design Considerations](#mcp-design-considerations)
1. [Running the Demos](#running-the-demos)
    1. [Everything Locally-Served](#everything-locally-served)
    1. [Connecting QAG MCP to Claude Desktop](#connecting-qag-mcp-to-claude-desktop)
1. [Extending QAG with Cohort Copilot](#extending-qag-with-cohort-copilot)
    1. [Making Minimal Changes](#making-minimal-changes)
1. [To Do](#to-do)

## Conceptual Reframing

This demo serves to modernize the Query Augmented Generation (QAG) framework using contemporary agentic concepts. The first iteration of QAG relied on a custom LLM to infer the intent of user queries. After categorizing the intent, a secondary model and static retrieval system were used to gather the entities required to query the GDC API. Manual routing and orchestration using the intent and entities was done to construct and execute the GDC API queries. Finally, the resulting information was parsed by a tertiary LLM for final return to the user. These steps are fragile and do not scale well: the custom model to infer user intent was trained on poorly generated, templated synthetic queries for a small subset of tasks; additional tasks require manual curation of entity maps as well as custom router logic; and tasks are hard-coded and result in code duplication without sufficient modularization.

Modern agentic LLMs with well-documented MCP tools are sufficiently powerful to simplify all of these steps. Given MCP tool descriptions, an LLM agent can determine which tools are most appropriate for the given user query. Additionally, the agent can dynamically extract the entitities from the user's query and correctly prepare them in the input format required by the tools. When multiple steps of API queries are potentially necessary, the agent can dynamically plan and execute multi-round tool calls to accomplish a complex task. Yet, for simpler tasks, the modular tool design allows for direct reuse without custom logic or implementation.

## Revamped MCP Architecture

The overall architecture of agentic QAG is diagramed in this figure:

![QAG v2 Agentic Architecture](figs/2026-02-23%20QAG%20Agent%20Demo.png)

### Guiding Use Cases

In the GDC-QAG paper, there are 6011 templated queries that are used for evaluation. These queries can be categorized and solved as follows:

| Type | Example | Solve Strategy |
|------|---------|----------------|
| SSM and CNV by Project | What percentage of cancers have simple somatic mutations and copy number variants in BRAF in the genomic data commons for NCI Cancer Model Development for the Human Cancer Model Initiative HCMI-CMDC project? | 1. Find cases C1 with SSM in gene<br>2. Find cases C2 with CNV in gene<br>3. Find cases P within project<br>4. Compute C12 = C1 ∩ C2<br>5. Compute Ret = C12 ∩ P |
| SSM by Project | How often is the BRAF V600E found in Skin Cutaneous Melanoma TCGA-SKCM project in the genomic data commons? | 1. Find cases C with SSM in gene resulting in AA change<br>2. Find cases P within project<br>3. Compute Ret = C ∩ P |
| CNV by Project | What is the frequency of somatic JAK2 heterozygous deletion in Acute Lymphoblastic Leukemia - Phase II TARGET-ALL-P2 project in the genomic data commons? | 1. Find cases C with CNV in gene with specific change type<br>2. Find cases P within project<br>3. Compute Ret = C ∩ P |
| MSI by Project | How common is microsatellite instability in Genomic Characterization CS-MATCH-0007 Arm S1 MATCH-S1 project cases in the genomic data commons? | 1. Find cases C with MSI<br>2. Find cases P within project<br>3. Compute Ret = C ∩ P |
| Co-occurrence by Project | What is the co-occurence frequency of somatic heterozygous deletions in CDKN2A and CDKN2B in the mesothelioma project TCGA-MESO in the genomic data commons? | 1. Find cases C1 with condition 1<br>2. Find cases C2 with condition 2<br>3. Find cases P within project<br>4. Compute C12 = C1 ∩ C2<br>5. Compute Ret = C12 ∩ P |

### Modularization and Reusable Components
The question types presented above can be solved with a minimal set of tools, specifically:
1. tool to compute case intersection
1. tools to retrieve cases:
    1. by project
    1. by SSM within a gene (optionally resulting in an AA change)
    1. by CNV within a gene (optionally with a specific change type)
    1. by MSI status

### MCP Design Considerations
When architecting the MCP tools, we consider the following design criteria and how these have been implemented:
* MCP servers should expose specialized, yet modular tools that LLM agents can dynamically use (and reuse) - e.g. a calculator MCP with arithmetic tools.
    * e.g. Project case retrieval has been disentagled from each tool and per-project cases with conditions can be computed using set intersection.
* To enable LLMs to effectively leverage MCP tools, there must be a strong natural language text interface between the LLM and the MCP server.
    * e.g. The documentation for the SSM tool parameter `aa_change` specifies that it is in HGVS format with an example.
* The crux of modern LLM agents is their ability to understand and plan around MCP server instructions, tool descriptions, tool parameter descriptions, tool return semantics, and **what to do when the tools fail**.
    * e.g. If a tool fails, we provide an error message that lets the LLM agent know how to recover from it by requerying a specific tool.
* However, LLMs still have (lots of) limitations, including non-natural language text, context window length, and potentially long context forgetting, so tools should avoid injecting text that may exacerbate these known LLM limitations.
    * e.g. Case sets are cached server side in a TTL cache, preventing context window clutter with lists of non-natural language case UUIDs.

Take a look at the [Extending QAG with Cohort Copilot](#extending-qag-with-cohort-copilot) section for an example of the modularity we've designed!

## Running the Demos

We demonstrate two ways of utilizing agentic QAG:

### Everything Locally-Served

This example requires a high-capacity GPU (at minimum an NVIDIA A100 80GB) to serve the reasoning model `gpt-oss-120b`. Less capable open-weight models may also be used but results do improve with stronger reasoning and multistep planning capabilities. For this method, we use the following tools:
* `openai/gpt-oss-120b` as our reasoning model
* `vLLM` to serves the LLM
* `FastMCP` to create our MCP server
* `pydantic-ai` to create our agent and to provide an AG-UI server
* `CopilotKit` to create our UI app server

#### Setup

Clone this repo and setup your environment. We use conda below but you can alternatively use any other virtual environment manager (eg `uv`) and install directly from the various requirement files. If you use something other than conda, make sure to use the same python version (`3.12.12`).

```bash
# general environment dependencies
conda create -f env.yaml
conda activate qag-agent-demo

# install dependencies for frontend app
cd frontend
npm install
cd ..

# install mcp server in editable mode
pip install -e .
```

#### Run Servers

Once you've setup and activated your environment, start the ollama, MCP, and agent UI servers before opening the agent UI in your browser:
```bash
# start the vLLM server, change the GPU index as needed
# you can also use docker to serve vllm though there are some FIPS issues if you need to care about those
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-120b   --served-model-name model   --enable-auto-tool-choice   --tool-call-parser openai

# start the MCP server (in a separate terminal window)
python -m qag_mcp.server -t streamable-http -p 8001

# start the agent AG-UI server (in a separate terminal window)
python agent/agent.py -p 8003 --mcp-url "http://localhost:8001/mcp" --llm-url "http://localhost:8000/v1"

# start the UI app server (in a separate terminal window)
cd frontend
AGENT_URL="http://localhost:8003" npm run dev -- -p 8002

# alternatively, you can build and serve the frontend app
cd frontend
npm run build
AGENT_URL="http://localhost:8003" npm run start -- -p 8002

# if you need to serve the frontend app behind a reverse proxy that does not strip the app prefix, you'll need to set the root path via NEXT_PUBLIC_BASE_PATH
AGENT_URL="http://localhost:8003/" npm run dev -- -p 8002

# note that if you're building the app, the base path must be set at both build and start time
NEXT_PUBLIC_BASE_PATH=/prefix npm run build
NEXT_PUBLIC_BASE_PATH=/prefix AGENT_URL="http://localhost:8003" npm run start -- -p 8002

# I'm not a software engineer so the frontend app was mostly vibe coded and based off this example:
# https://github.com/CopilotKit/with-pydantic-ai

# go to http://localhost:8002 in a browser and query away
```

If you are running the servers on a remote host, you can use ssh tunneling to forward the remote port of the app server to your local so you can access the UI on your local.

#### Local LLM Agent Limitations

Open-weight models tend to still be behind frontier models (like those that back the Claudes, ChatGPTs, and Geminis). Of course, the local agent could switch backends but we tend to want to use a local LLM with a local agent. One salient example of this difference in capability that came up in testing was letting the model do math directly in natural language. It was okay, but it would sometimes prematurely terminate. Our solution was to just add basic arithmetic tools directly to the agent which seems to have resolved the math computation limitation. Though not quantitative, we have found `gpt-oss-120b` to perform well in powering our agentic framework.

### Connecting QAG MCP to Claude Desktop

If you don't have access to a high-capacity GPU, you can still test out the demo by connecting the MCP server to a frontier-model provider (honestly, these resutls tend to be stronger). This section specifically details how to test out the MCP tools using Claude Desktop. At the time of writing, you can do all these steps with just a free-tier account.

1. Download and install Claude Desktop (https://claude.com/download)
1. Download and install `uv` (https://docs.astral.sh/uv/#installation)
1. Clone this repository and setup your virtual environment:
    ```
    git clone git@github.com:StevenSong/qag-agent-demo.git
    cd qag-agent-demo
    uv venv --python 3.12.12
    uv pip install -r requirements-mcp.txt
    uv pip install .
    ```
1. Open Claude Desktop, go to Settings > Developer > Edit Config, and open the file `claude_desktop_config.json` in your favorite text editor
1. Add the following section to your JSON config file, note that you should provide absolute paths for `uv` and `qag-agent-demo`:
    ```
    "mcpServers": {
        "gdc": {
            "command": "/full/path/to/uv",
            "args": [
                "--directory",
                "/full/path/to/qag-agent-demo",
                "run",
                "-m",
                "qag_mcp.server",
                "-t",
                "stdio"
            ]
        }
    }
    ```
1. Restart Claude Desktop (must fully close the application to reload the config)
1. Verify that the `gdc` MCP server has started by checking the enabled "Connectors" (from the main prompt UI, click the `+` sign in the lower left, Connectors submenu)
1. Try the example queries (you'll need to approve the tool usage through the UI)

## Extending QAG with Cohort Copilot

One of the key aspects of the redesign of QAG is its modularity. The `get_cases_by_project` tool simply queries the GDC's `/cases` endpoint with a manually constructed filter on the project ID. However, we already have a tool that makes arbitrary filters for the `/cases` endpoint: [GDC Cohort Copilot](https://github.com/uc-cdis/gdc-cohort-copilot). We demonstrate how we can use cohort copilot as a dropin replacement and generalization for the case retrieval. To that end, we provide a command line toggle in the MCP server to enable cohort copilot.

**NOTE:** GDC Cohort Copilot is undergoing major revisions (just like QAG)! The first version of cohort copilot utilized an ultralightweight GPT2 model that can run on CPU-only. We use that version here for this demo, however this requires some extra dependencies that should be installed:

```bash
pip install -r requirements-cohort-copilot.txt
```

Once installed, just use the `--use-cohort-copilot` flag when starting up the MCP server (either with locally hosted agents or in your MCP server config for external tools). Take a look at the minor changes between `get_cases_by_project` and `get_cases_by_cohort_description` in `qag_mcp/server.py`. These minor changes to enable arbitrary cohort descriptions allow answering extensively more questions while reusing almost entirely the same tools/code! For example, try out this query:

> for patients with mutation in KRAS, what is the difference in prevalence between male vs female patients?

Note that if you do have a CUDA enabled GPU on your machine, the MCP server will try to put the cohort copilot model onto the GPU (though GPU is not necessary for the cohort copilot model). The model is relatively tiny but if you need to be aware of GPU resources, or if you need to control which GPU the model uses on a multi-gpu system, use the standard `CUDA_VISIBLE_DEVICES` env var while launching.

### Making Minimal Changes

When making changes to an MCP toolset, consider the [design principles](#mcp-design-considerations) we discussed. In a nutshell, when we change the case retrieval tool from project filtering to arbitrary cohort descriptions, we needed to update the following things:
* the actual filter logic
* the tool description
* the parameter description
* the error handling

## To Do

* Add `genes` endpoint, consider these examples:
    * What is the variance of MYC expression across projects?
    * Is PD-L1 (CD274) higher in male vs female patients?
    * Is MYC expression higher in smokers?
    * Does EGFR expression differ by stage?
    * Is KRAS more highly expressed in pancreatic vs lung cancer?
* Add `survival` endpoint
