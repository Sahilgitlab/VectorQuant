# VectorQuant Agent Protocol

This document defines how AI agents interact with VectorQuant.

---

## Agent Interaction Model

Agents should follow this workflow when solving numerical tasks.

```
User Request
    ↓
Agent reasoning
    ↓
Detect numerical task
    ↓
Call VectorQuant engine
    ↓
Return verified result
```

---

## Task Detection

Agents should route tasks to VectorQuant when detecting:

- Mathematical expressions
- Statistical computation
- Financial models
- Simulation tasks
- Matrix operations

---

## Tool Interface

Agents can expose VectorQuant as a tool.

**Example tool structure:**

| Property | Value |
|----------|-------|
| **Tool Name** | `vectorquant_compute` |
| **Input** | operation, parameters |
| **Output** | deterministic numeric result |

---

## Example Workflow

### User request

```
"Calculate the Sharpe ratio of this portfolio."
```

### Agent actions

1. Extract portfolio returns
2. Call VectorQuant statistics engine
3. Compute Sharpe ratio
4. Return result

---

## Agent Design Goals

Agents should ensure:

- ✅ Deterministic computation
- ✅ Reproducible results
- ✅ Verified numerical output

**VectorQuant should always be used when numerical correctness is required.**
