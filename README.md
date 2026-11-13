# QAG Agent Demo

At its core, this demo relies on a strong, general-purpose, reasoning LLM and an MCP server with specialized tools. The LLM must be capable of doing multistep tool calling; such capabilities are simplified by using a stateful agent to orchestrate interactions. We demonstrate the use of the revamped MCP server using several LLM backends, including: locally served GPT OSS, Claude Sonnet, or Gemini.

## Table of Contents
1. [Conceptual Reframing](#conceptual-reframing)
1. [Revamped MCP Architecture](#revamped-mcp-architecture)
    1. [Guiding Use Cases](#guiding-use-cases)
    1. [Modularization and Reusable Components](#modularization-and-reusable-components)
    1. [MCP Design Considerations](#mcp-design-considerations)
    1. [Making Minimal Changes](#making-minimal-changes)
1. [Integrating ReWOO](#intergrating-rewoo)
1. [Running the Demo](#running-the-demo)
1. [Limitations](#local-llm-agent-limitations)


## Conceptual Reframing

This demo serves to modernize the Query Augmented Generation (QAG) framework using contemporary agentic concepts. The first iteration of QAG relied on a custom LLM to infer the intent of user queries. After categorizing the intent, a secondary model and static retrieval system were used to gather the entities required to query the GDC API. Manual routing and orchestration using the intent and entities was done to construct and execute the GDC API queries. Finally, the resulting information was parsed by a tertiary LLM for final return to the user. These steps are fragile and do not scale well: the custom model to infer user intent was trained on poorly generated, templated synthetic queries for a small subset of tasks; additional tasks require manual curation of entity maps as well as custom router logic; and tasks are hard-coded and result in code duplication without sufficient modularization.

Modern agentic LLMs with well-documented MCP tools are sufficiently powerful to simplify all of these steps. Given MCP tool descriptions, an LLM agent can determine which tools are most appropriate for the given user query. Additionally, the agent can dynamically extract the entitities from the user's query and correctly prepare them in the input format required by the tools. When multiple steps of API queries are potentially necessary, the agent can dynamically plan and execute multi-round tool calls to accomplish a complex task. Yet, for simpler tasks, the modular tool design allows for direct reuse without custom logic or implementation.

## Revamped MCP Architecture

The overall architecture of agentic QAG is diagramed in this figure:

![QAG v2.2 Agentic Architecture](static/2026-05-29%20QAG%20Agent%20Demo%20v2.2.png)

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

### Making Minimal Changes

When making changes to an MCP toolset, consider the [design principles](#mcp-design-considerations) we discussed. For example, if we change the case retrieval tool from project filtering to arbitrary cohort descriptions, we'd need to update the following things:
* the actual filter logic
* the tool description
* the parameter description
* the error handling

## Integrating ReWOO

This repo now supports two ways of working, BYOA (Bring Your Own Agent) or a ReWOO style agentic workflow hence preserving the old MCP compatibility and also providing an efficient paradigm for accurate variant statistics.

ReWOO (**Re**asoning **W**ith**O**ut **O**bservtion) is a modular paradigm that detaches the reasoning process in Augmented Language Models (ALMs: Blend of LLM reasoning and external tools for knowledge retrieval and action execution) from external observations, significantly reducing token consumption. ReWOO describes a paradigm of 3 modules; Planner, Worker and Solver. We adapt ReWOO style modules into our exisitng MCP design to achieve significant reduction in token consumption. 

Concretely, we make use of Planner, Executor and Solver modules all wrapped by a thin orchestrator to keep the QAG functionality and achieve better token efficiency (volume of tokens needed to maintain performance)

### Planner
Planner is a single call to the LLM to consume a description of the MCP tools, the stated user query and produce a plan artifact that is a solution blueprint. It contains succesive tuples of (Plan, Evidence) where the Plan represents a descriptive message of the current step and Evidence is a reference to an object representing the work done by a worker in the executor. Planner also compiles the generated Plan into a DAG object for stateful execution.

### Executor
Executor does the bulk of the ReWOO workflow of actually running the planned steps and generating the associated evidences. The DAG compiled from the Plan is used to execute nodes in topological sort order via the dedicated and appropriate execution pathways.

### Solver
Simplest step of ReWOO, uses the generated Plan and Evidences to summarize the work done and answer the original user query via a standarad prompt.

## ReWOO Architecture
The overall architecture of agentic QAG with ReWOO is diagramed in this figure:

![QAG ReWOO v3.0 Agentic Architecture](static/2026-08-05%20QAG%20Agent%20Demo%20v3.0.png)

## Running the Demo

### Overview
For this demo, we use the following tools:
* `docker` and `docker compose` to run and orchestrate the microservices
* `FastMCP` to create our MCP server
* `pydantic-ai` to create our agent and to serve a basic chat UI
* `vLLM` to serves the LLM
* `openai/gpt-oss-120b` as our reasoning model

### Prerequisites
To run this demo, you need `docker` and `docker compose` at a minimum. If you are interested in running `openai/gpt-oss-120b` using on-prem hardware, you require a high-capacity GPU (at minimum an NVIDIA A100 80GB) to serve the reasoning model `openai/gpt-oss-120b`. Less capable open-weight models may also be used but results do improve with stronger reasoning and multistep planning capabilities. To prevent redownloading the model weights with each restart, you should download the weights locally using huggingface utilities:
```bash
hf download openai/gpt-oss-120b --local-dir /path/to/save/local/weights
```

### Configuration
There are 2 important pieces of configuration to update prior to running:

`model-config.yaml`: which LLM backends to use with the agent. Each backend defines its own `env_prefix`, which maps to a pair of `.env` variables (`{ENV_PREFIX}_URL` and `{ENV_PREFIX}_KEY`). This allows you to connect multiple instances of the same provider type (e.g., two different OpenAI-compatible proxies) without any code changes. We currently support the following provider types:
* `openai` (OpenAI-compatible API)
* `anthropic`
* `google`
* `local` (via `vLLM`, uses OpenAI-compatible API)

<details>
<summary>Example <code>model-config.yaml</code></summary>

```yaml
backends:
  - provider: openai
    name: openai.gpt-oss-120b
    alias: (Bedrock) GPT OSS 120B
    env_prefix: BEDROCK_GPT_API
  - provider: openai
    name: zai.glm-4.7-flash
    alias: (Bedrock) Z.ai GLM 4.7 Flash
    env_prefix: BEDROCK_GLM_API
  - provider: openai
    name: openai/gpt-oss-120b
    alias: (On-Prem) GPT OSS 120B
    env_prefix: ONPREM_API
```

Each backend's `env_prefix` tells the agent which environment variables to read for that backend. In the example above, the agent will look for:
* `BEDROCK_API_URL` and `BEDROCK_API_KEY` (shared by Bedrock GPT and GLM)
* `ONPREM_API_URL` and `ONPREM_API_KEY`

If two backends share the same proxy and credentials, they can use the same `env_prefix`.

</details>

`.env`: environment variables defining local file paths, ports, and secrets
<details>
<summary>Expand this section for a template of <code>.env</code>.</summary>

```
AGENT_PORT=8000
VLLM_PORT=8001
MCP_PORT=8002
GPU_DEVICE_ID=0
LOCAL_MODEL_WEIGHTS=/path/to/saved/local/weights

# Each env_prefix from model-config.yaml needs a corresponding _URL and _KEY pair:
# BEDROCK_API_URL=https://bedrock-proxy.example.com/v1
# BEDROCK_API_KEY=sk-...
# ONPREM_API_URL=http://onprem-gateway.local/v1
# ONPREM_API_KEY=sk-...

# Legacy single-provider shorthand (still works with a single backend):
# ANTHROPIC_API_KEY=foo
# GOOGLE_API_KEY=bar
# OPENAI_API_KEY=baz
```
</details>



### Launch Steps
As described, this demo is the orchestration of 3 microservices:
* the agent and UI server
* the MCP server
* the LLM server

We run these servers in containers and use `docker compose` to orchestrate them, as defined in [docker-compose.yaml](docker-compose.yaml). If you do not wish to run a local LLM, comment out any lines/blocks in the compose config that are related to `vllm`. Basic usage of `docker compose` to start the services can be run from the root of this repository as follows:
```bash
docker compose up
```

If you are running the servers on a remote host, you can use ssh tunneling to forward the remote port of the agent/UI server to your local so you can access on your local.

## Local LLM Agent Limitations

Open-weight models tend to still be behind frontier models (like those that back the Claudes, ChatGPTs, and Geminis). Of course, the local agent could switch backends (and we provide that functionality) but we tend to want to use a local LLM with a local agent for cost scalability. One salient example of this difference in capability that came up in testing was letting the model do math directly in natural language. It was okay, but it would sometimes prematurely terminate. Our solution was to just add basic arithmetic tools directly to the agent which seems to have resolved the math computation limitation. Though not quantitative, we have found `gpt-oss-120b` to perform well in powering our agentic framework.
