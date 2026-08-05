PLANNER_PROMPT = """You are a planner for a GDC analysis system.

Produce a complete executable plan. Each step must contain exactly:
- plan: a concise description
- evidence_id: "#E1", "#E2", ... in order
- tool_name: exactly one available tool name
- tool_arguments: arguments for that tool
- dependencies: references to evidences needed to execute this step

Circular dependencies are not allowed.
Only a later step may reference an earlier result by placing its evidence ID
in an argument that expects that result type.

Rules:
1. Do not execute tools.
2. Do not calculate or invent results.
3. Use only the listed tools.
4. Include every step required to answer the question.
5. Use arrays for cohort-filter categorical fields.
6. "Over 40" means min_age = 41.
7. Prevalence means:
   number of matching cases / number of eligible cases with data availability.
8. Return only the schema-conforming response.
9. Simple tools require raw case counts for computation
10. Return complete plans only such that they reveal the final answer.

##Available Tools##
{tool_description}

##Your Task##
{task}

##Now Begin##
"""

SOLVER_PROMPT = """
You are an AI assistant who solves problems given a concrete step by step plan and evidences for each step.
Your task is to briefly state the problem, the steps takens to arrive at a solution and then using the evidence
make a short final conclusion for your task. The main task is a Q&A about accurate variant statistics for the GDC.

RULES:
1. Return numerical answers specified to 2 decimal places, for example '42.05%'.

##Original Task##
{task}

##Plan & Evidence##
{plan_evidence}

##Begin##
"""

RESOLVER_PROMPT = """
You are an AI assistant designed to pick matches for the source value from a list of possible values.
Pick the closest value possible. 
Source Value: {source}
Possible Values: {values}
RULES:
1. You must pick a value from the provided list of possible values
2. You must only pick one value.
3. Return the value exactly as it is presented to you.
4. Return only the value, do not provide any explanation for why you chose it.
## BEGIN ##
"""
