---
name: context-engineering
description: Covers context engineering principles and practices for designing and building autonomous AI agents. Explains how to manage agent memory (short-term, long-term, and episodic), configure and connect tools, and apply architectural patterns for single- and multi-agent systems. Use when the user asks about building AI agents, designing agent architectures, managing context windows, orchestrating multi-agent workflows, structuring agent memory, or implementing autonomous AI systems with tool use and retrieval.
---

# Context Engineering

Context engineering is the discipline of designing, structuring, and managing the information an AI agent receives at each step so it can reason and act effectively. It covers three core areas: **Memory**, **Tools**, and **Patterns**.

---

## Memory

Three memory types govern what information persists across steps and sessions:

- **Short-term (in-context)** — Active context window: current task state, recent tool outputs, conversation history. Prune aggressively to avoid overflow.
- **Long-term (external)** — Retrieved on demand from vector stores or key-value stores. Retrieval must be query-scoped — fetch only what is directly relevant to the current step.
- **Episodic** — Structured logs of past agent runs. Use to surface what has already been tried and avoid repeating failed approaches.

**Memory design workflow:**
1. Identify what the agent needs at each reasoning step.
2. Classify each piece: short-term (fits in context), long-term (needs retrieval), or episodic (derived from past runs).
3. Define clear read/write boundaries — when is memory updated, and when is it read?
4. Validate retrieved context is scoped to the current query; never inject broad, unfiltered memory dumps.

**Scoped retrieval — using ChromaDB + OpenAI:**
```python
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_collection("agent_docs")

def build_agent_context(system_prompt: str, session_id: str, task: str) -> str:
    # Only retrieve chunks relevant to the current step, not the full history
    results = collection.query(
        query_texts=[task],
        n_results=5,
        where={"session_id": session_id}
    )
    chunks = "\n\n".join(results["documents"][0])
    return f"{system_prompt}\n\n## Retrieved Context\n{chunks}\n\n## Current Task\n{task}"
```

---

## Tools

**Key practices:**
- **Define precise tool signatures** — Name, description, and parameter schema must be unambiguous. Vague descriptions cause misuse or skipping.
- **Limit the active tool set** — Present only tools relevant to the current task; large undifferentiated lists increase wrong-tool selection.
- **Handle tool outputs explicitly** — Specify how the agent interprets outputs, including error states and empty results.
- **Chain tools deliberately** — Define the expected call sequence and what intermediate state passes between steps.

**Validation checkpoint:** After configuring tools, verify (a) each description unambiguously distinguishes it from similar tools, and (b) the agent can complete the target task using only the tools provided — no missing capabilities, no redundant overlaps.

**Precise tool definition — OpenAI function-calling format:**
```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Searches internal documentation. "
                "Use ONLY for policy or reference lookups — "
                "do NOT use for live data or calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_calculation",
            "description": (
                "Executes arithmetic or statistical calculations. "
                "Use for any numeric computation — "
                "do NOT use for text lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the refund policy, and what is 15% of $240?"}],
    tools=tools,
    tool_choice="auto"
)
```

---

## Patterns

> For full implementation sketches of each pattern, see `PATTERNS.md`.

- **ReAct (Reason + Act)** — Agent alternates reasoning steps with tool calls. Use when tasks require multi-step decision-making with intermediate results.
- **Plan-then-Execute** — Agent produces a full plan before acting. Use when scope is fully known upfront; reduces mid-task drift.
- **Multi-agent orchestration** — Coordinator delegates subtasks to specialised sub-agents. Use when subtasks are independent, parallelisable, or require different tool sets.
- **Reflection / Self-critique** — Agent reviews its own output before finalising. Use when output quality is critical (e.g., code correctness, factual accuracy).

**Choosing a pattern:**
1. Decomposable into parallel subtasks? → Multi-agent orchestration.
2. Full plan knowable upfront? → Plan-then-Execute.
3. Requires iterative tool use? → ReAct.
4. Output quality is primary risk? → Add a Reflection step to any of the above.

---

## Common Pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| Agent loops or repeats failed tool calls | Episodic memory not wired in; agent cannot see prior attempts | Add episodic store read at session start; include recent run outcomes in context |
| Agent picks the wrong tool | Tool descriptions are ambiguous or overlap | Rewrite descriptions with explicit "Use ONLY for…" / "do NOT use for…" guards |
| Context window overflow mid-task | Short-term memory grows unbounded | Prune conversation history to a rolling window; move stable info to long-term retrieval |
| Retrieved context is off-topic | Retrieval query is too broad or unfiltered | Scope queries to `current_task_description`; apply metadata filters (e.g., `session_id`, `doc_type`) |
| Agent drifts from the original goal | No upfront plan; ReAct loop re-reasons scope each step | Switch to Plan-then-Execute; lock the plan before any tool calls |
| Poor output quality / hallucinations | No quality gate | Add a Reflection pass before returning the final answer |
| Multi-agent results are inconsistent | Sub-agents share no common context | Pass a shared `task_brief` and `constraints` object to every sub-agent at spawn time |

**Diagnosing failures — checklist:**
1. Is the agent missing information it needs? → Fix the memory layer (add retrieval or widen scope).
2. Is the agent retrieving irrelevant information? → Tighten retrieval filters or reduce `top_k`.
3. Did the agent call the wrong tool or skip a tool? → Tighten tool descriptions; reduce the active tool set.
4. Did the agent drift from the goal? → Switch or add a planning step.
5. Did the agent produce low-quality output? → Add or strengthen the Reflection pass.

---

## Complete Example: Agent Configuration

The following shows memory, tools, and pattern selection working together for a research-and-summarise task:

```python
import asyncio
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.HttpClient(host="localhost", port=8000)
docs_collection = chroma.get_collection("agent_docs")
episodic_collection = chroma.get_collection("episodic_store")

def get_prior_runs(task_type: str, limit: int = 3) -> str:
    results = episodic_collection.query(
        query_texts=[task_type], n_results=limit
    )
    return "\n\n".join(results["documents"][0]) if results["documents"] else ""

def get_domain_docs(query: str, top_k: int = 5) -> str:
    results = docs_collection.query(query_texts=[query], n_results=top_k)
    return "\n\n".join(results["documents"][0]) if results["documents"] else ""

def save_episodic(task: str, outcome: str) -> None:
    episodic_collection.add(
        documents=[outcome],
        metadatas=[{"task": task, "task_type": "research"}],
        ids=[f"run-{hash(task)}"]
    )

# 1. Memory — combine scoped retrieval with episodic context
user_query = "Summarise the competitive landscape for vector databases."
system_prompt = "You are a research assistant. Be concise and cite sources."
prior_runs  = get_prior_runs(task_type="research", limit=3)
domain_docs = get_domain_docs(query=user_query, top_k=5)
context = f"{system_prompt}\n\n## Past Runs\n{prior_runs}\n\n## Domain Docs\n{domain_docs}"

# 2. Tools — only what this task needs
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Searches internal documentation for policy or reference material.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 3}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetches the content of a public URL. Use for live web sources only.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]
            }
        }
    }
]

# 3. Pattern — ReAct via iterative tool-call loop (max 10 steps)
messages = [
    {"role": "system", "content": context},
    {"role": "user", "content": user_query}
]
for _ in range(10):
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            tool_result = dispatch_tool(tc.function.name, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})
    else:
        draft = msg.content
        break

# 4. Reflection — quality check before returning
critique_prompt = (
    f"Review the following research summary for accuracy, completeness, and absence of hallucinations.\n\n"
    f"Summary:\n{draft}\n\nIf there are issues, list them. Otherwise respond 'LGTM'."
)
critique = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": critique_prompt}]
).choices[0].message.content

if critique.strip() != "LGTM":
    revise_prompt = f"Revise the summary based on this feedback:\n{critique}\n\nOriginal:\n{draft}"
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": revise_prompt}]
    ).choices[0].message.content
else:
    final = draft

save_episodic(task=user_query, outcome=final)  # write back for future runs
print(final)
```

---

## General Workflow for Context Engineering

1. **Define the task scope** — What is the agent trying to accomplish? What are the success criteria?
2. **Design the memory architecture** — Which memory types are needed? What is retrieved, when, and how?
3. **Configure the tool set** — Which tools are required? Are descriptions precise and non-overlapping?
4. **Select an architectural pattern** — Which pattern fits the task structure?
5. **Test with minimal context** — Start with the smallest viable context and add information only when demonstrably needed.
6. **Iterate on failures** — When the agent fails, use the Common Pitfalls checklist to diagnose whether the cause is missing context, wrong memory retrieval, a misconfigured tool, or a pattern mismatch, then fix the specific layer.
