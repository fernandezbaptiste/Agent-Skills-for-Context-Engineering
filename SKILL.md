---
name: context-engineering-collection
description: Design agent memory systems, implement tool-calling patterns, structure multi-agent communication, and debug context-related failures in LLM agent workflows. Use when building or optimizing AI agent or LLM agent architectures—including ReAct-style agents, multi-agent pipelines, agent tools, agent prompts, and agent memory—or when troubleshooting agent loops, context overflow, or degraded agent performance in production systems.
---

# Agent Skills for Context Engineering

This collection provides structured guidance for building production-grade AI agent systems through effective context engineering.

## When to Activate

Activate these skills when:
- Building new agent systems from scratch
- Optimizing existing agent performance
- Debugging context-related failures
- Designing multi-agent architectures
- Creating or evaluating tools for agents
- Implementing memory and persistence layers

## Skill Map

### Foundational Context Engineering

**Understanding Context Fundamentals**
See → [context-fundamentals](skills/context-fundamentals/SKILL.md)

**Recognizing Context Degradation**
See → [context-degradation](skills/context-degradation/SKILL.md)

### Architectural Patterns

**Multi-Agent Coordination**
Choose among three dominant patterns based on control requirements: supervisor/orchestrator architectures for centralized control, peer-to-peer swarm architectures for flexible handoffs, and hierarchical structures for complex task decomposition. Design sub-agents to isolate context rather than simulate organizational roles.

Example — supervisor pattern with isolated sub-agent contexts:
```python
# Supervisor dispatches tasks to sub-agents, each with a clean context window
def supervisor_dispatch(task, available_agents):
    plan = supervisor_llm.plan(task)  # high-level decomposition
    results = {}
    for step in plan.steps:
        agent = select_agent(step, available_agents)
        # Each sub-agent receives only the context it needs
        sub_context = build_minimal_context(step, prior_results=results)
        results[step.id] = agent.run(sub_context)
    return supervisor_llm.synthesize(plan, results)
```

See → [multi-agent-patterns](skills/multi-agent-patterns/SKILL.md)

**Memory System Design**
Select memory tiers based on retrieval needs: use scratchpads for fast ephemeral state, vector stores for semantic retrieval across sessions, and knowledge graphs when relationship structure must be preserved. Use the file-system-as-memory pattern to enable just-in-time context loading and avoid unnecessary token consumption.

Example — initializing a tiered memory system (scratchpad + vector store):
```python
# Tier 1: in-session scratchpad (fast, ephemeral)
scratchpad = {}

# Tier 2: filesystem (persistent, structured)
def persist_to_fs(key, value, workspace="./agent_workspace"):
    path = f"{workspace}/{key}.json"
    write_file(path, json.dumps(value))
    return path

# Tier 3: vector store (semantic retrieval across sessions)
def store_with_embedding(text, metadata, vector_store):
    embedding = embed(text)
    vector_store.upsert(embedding, metadata)

# Load only what's needed at inference time
def load_context_for_task(task_id, vector_store):
    relevant_docs = vector_store.query(task_id, top_k=5)
    scratchpad_path = f"./agent_workspace/{task_id}_scratch.json"
    scratch = read_file(scratchpad_path) if file_exists(scratchpad_path) else {}
    return {"docs": relevant_docs, "scratch": scratch}
```

See → [memory-systems](skills/memory-systems/SKILL.md)

**Filesystem-Based Context**
Use the filesystem as a single interface for storing, retrieving, and updating effectively unlimited context. Key patterns: scratch pads for tool output offloading, plan persistence for long-horizon tasks, sub-agent communication via shared files, and dynamic skill loading. Prefer `ls`, `glob`, `grep`, and `read_file` for targeted context discovery over semantic search on structural queries.

Example — targeted context discovery before loading files into context:
```bash
# Locate relevant files without reading everything
ls -la ./agent_workspace/
grep -r "task_id=abc123" ./scratchpads/ --include="*.json" -l
glob "./plans/**/*.md"
# Then read only the matched files
read_file ./plans/task_abc123_plan.md
```

Example — offloading verbose tool output to a scratchpad:
```python
# Instead of returning 10k tokens of API response into context:
result = call_external_api(query)
scratchpad_path = f"./scratchpads/{task_id}_api_result.json"
write_file(scratchpad_path, json.dumps(result))
# Return only a reference
return f"API result saved to {scratchpad_path}. Key fields: status={result['status']}, count={result['count']}"
```

See → [filesystem-context](skills/filesystem-context/SKILL.md)

**Hosted Agent Infrastructure**
Background coding agents run in remote sandboxed environments. Key patterns include pre-built environment images, warm sandbox pools, filesystem snapshots for persistence, and multiplayer support. Critical optimizations: allow file reads before git sync completes (blocking only writes), predictive sandbox warming, and self-spawning agents for parallel task execution.

See → [hosted-agents](skills/hosted-agents/SKILL.md)

**Tool Design Principles**
Treat tools as contracts between deterministic systems and non-deterministic agents. Follow the consolidation principle (prefer single comprehensive tools over multiple narrow ones), return contextual information in errors, support response format options for token efficiency, and use clear namespacing.

See → [tool-design](skills/tool-design/SKILL.md)

### Operational Excellence

**Context Compression**
When agent sessions exhaust memory, target tokens-per-task rather than tokens-per-request. Use structured summarization with explicit sections for files, decisions, and next steps — this preserves more useful information than aggressive compression. Prioritize artifact trail integrity, which is the weakest dimension across most compression methods.

See → [context-compression](skills/context-compression/SKILL.md)

**Context Optimization**
Apply compaction (summarizing context near limits), observation masking (replacing verbose tool outputs with references), prefix caching (reusing KV blocks across requests), and strategic context partitioning (splitting work across sub-agents with isolated contexts).

See → [context-optimization](skills/context-optimization/SKILL.md)

**Evaluation Frameworks**
Use multi-dimensional rubrics covering factual accuracy, completeness, tool efficiency, and process quality. Apply LLM-as-judge for scalability, human evaluation for edge cases, and end-state evaluation for agents that mutate persistent state.

See → [evaluation](skills/evaluation/SKILL.md)

### Development Methodology

**Project Development**
Begin with task-model fit analysis: validate through manual prototyping that a task is well-suited for LLM processing before building automation. Follow staged, idempotent pipeline architectures (acquire, prepare, process, parse, render) with filesystem state management for debugging and caching. Start with minimal architecture and add complexity only when proven necessary.

See → [project-development](skills/project-development/SKILL.md)

## Troubleshooting Context-Related Failures

Use this workflow when an agent loop is degrading, stalling, or producing incorrect results:

**Step 1 — Identify the failure mode**
- [ ] Agent loop not terminating → likely context poisoning or missing stop condition; check [context-degradation](skills/context-degradation/SKILL.md)
- [ ] Outputs degrading over long sessions → likely lost-in-middle or context overflow; check [context-compression](skills/context-compression/SKILL.md)
- [ ] Agent ignoring relevant information → check retrieval pipeline and [memory-systems](skills/memory-systems/SKILL.md)
- [ ] Tool call errors compounding → check error propagation in [tool-design](skills/tool-design/SKILL.md)

**Step 2 — Measure context health**
```python
# Log context size and composition at each agent step
def log_context_health(context, step_id):
    total_tokens = count_tokens(context)
    breakdown = {
        "system": count_tokens(context.system),
        "history": count_tokens(context.history),
        "tool_outputs": count_tokens(context.tool_outputs),
        "retrieved_docs": count_tokens(context.retrieved_docs),
    }
    print(f"[step={step_id}] total={total_tokens} | {breakdown}")
    return total_tokens > CONTEXT_LIMIT * 0.8  # True = compression needed
```

**Step 3 — Apply targeted remediation**
- High token count in `tool_outputs` → apply observation masking (offload to scratchpad)
- High token count in `history` → apply structured compaction before limit is hit
- High token count in `retrieved_docs` → tighten retrieval query or reduce `top_k`

**Step 4 — Validate fix**
- Re-run the failing task end-to-end and confirm output quality
- Measure tokens-per-task before and after remediation
- For persistent state mutations, use end-state evaluation (see [evaluation](skills/evaluation/SKILL.md))

## Practical Guidance

Each skill can be used independently or in combination. Start with fundamentals to establish context management mental models. Branch into architectural patterns based on your system requirements. Reference operational skills when optimizing production systems.

The skills are platform-agnostic and work with Claude Code, Cursor, or any agent framework that supports custom instructions or skill-like constructs.

## References

Internal skills in this collection:
- [context-fundamentals](skills/context-fundamentals/SKILL.md)
- [context-degradation](skills/context-degradation/SKILL.md)
- [context-compression](skills/context-compression/SKILL.md)
- [multi-agent-patterns](skills/multi-agent-patterns/SKILL.md)
- [memory-systems](skills/memory-systems/SKILL.md)
- [tool-design](skills/tool-design/SKILL.md)
- [filesystem-context](skills/filesystem-context/SKILL.md)
- [hosted-agents](skills/hosted-agents/SKILL.md)
- [context-optimization](skills/context-optimization/SKILL.md)
- [evaluation](skills/evaluation/SKILL.md)
- [project-development](skills/project-development/SKILL.md)

External references:
- Research on attention mechanisms and context window limitations
- Production experience from leading AI labs on agent system design
- Framework documentation for LangGraph, AutoGen, and CrewAI
