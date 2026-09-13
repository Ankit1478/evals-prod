# Evals-Prod — Production Evaluation System for AI Agents & Tools

A harness that measures whether an AI agent did the right thing — not whether it
sounded like it did.

**Status:** planning complete, implementation not started.
**Owner:** Ankit Raj
**Last updated:** 2026-09-11

---

## Table of contents

1. [What this is and what it is not](#1-what-this-is-and-what-it-is-not)
2. [Design principles](#2-design-principles)
3. [Repository structure](#3-repository-structure)
4. [Core data model](#4-core-data-model)
5. [Adapters — the system under test](#5-adapters--the-system-under-test)
6. [Environment — stateful mock backend](#6-environment--stateful-mock-backend)
7. [Graders](#7-graders)
8. [Harness — the runner](#8-harness--the-runner)
9. [Statistics and the release gate](#9-statistics-and-the-release-gate)
10. [Store and run artifacts](#10-store-and-run-artifacts)
11. [Reporting](#11-reporting)
12. [The `eval.yaml` spec](#12-the-evalyaml-spec)
13. [CLI surface](#13-cli-surface)
14. [Testing the measurement system](#14-testing-the-measurement-system)
15. [Tech stack](#15-tech-stack)
16. [Reusing LLM_AS_JUDGE](#16-reusing-llm_as_judge)
17. [Build plan — 10 weeks](#17-build-plan--10-weeks)
18. [Phase 2 — the platform (overview)](#18-phase-2--the-platform-overview)
19. [CI/CD integration](#19-cicd-integration)
20. [Conventions and hard rules](#20-conventions-and-hard-rules)
21. [Known concept gaps to close](#21-known-concept-gaps-to-close)
22. [Mapping to the 228-topic handbook](#22-mapping-to-the-228-topic-handbook)
23. [Risks and pitfalls](#23-risks-and-pitfalls)
24. [Definition of done](#24-definition-of-done)
25. [Day 1 checklist](#25-day-1-checklist)
26. [Environments and deployment topology](#26-environments-and-deployment-topology)
27. [Phase 2 — persistence and API](#27-phase-2--persistence-and-api)
28. [Phase 2 — OpenTelemetry trace ingestion](#28-phase-2--opentelemetry-trace-ingestion)
29. [Phase 2 — online evaluation in production](#29-phase-2--online-evaluation-in-production)
30. [Phase 2 — review queue and the data flywheel](#30-phase-2--review-queue-and-the-data-flywheel)
31. [Phase 2 — dashboard](#31-phase-2--dashboard)
32. [Phase 3 — multi-tenant platform](#32-phase-3--multi-tenant-platform-only-if-needed)
33. [Security and threat model](#33-security-and-threat-model)
34. [Data governance, privacy, retention](#34-data-governance-privacy-retention)
35. [Cost model and budget control](#35-cost-model-and-budget-control)
36. [Operating the eval system](#36-operating-the-eval-system)
37. [Runbooks](#37-runbooks)
38. [Release management of the harness itself](#38-release-management-of-the-harness-itself)
39. [Team, ownership, and rituals](#39-team-ownership-and-rituals)
40. [Rollout: shadow → advisory → enforced](#40-rollout-shadow--advisory--enforced)
41. [Full timeline to production](#41-full-timeline-to-production)
42. [Program success metrics](#42-program-success-metrics)
43. [Documentation deliverables](#43-documentation-deliverables)
44. [Open decisions](#44-open-decisions)

---

## 1. What this is and what it is not

### The problem

Standard LLM eval frameworks grade **outputs**. Agents fail in **trajectories**.
Teams grading only final output measure pass rates 20–40% higher than full
trajectory evaluation reveals. An agent that says *"I've cancelled order 123"*
while having cancelled order 456 passes an output grader and fails reality.

### What this system does

Runs a versioned set of cases against an agent, N trials each, inside an
isolated stateful environment; captures the full trajectory (every tool call,
argument, result, and state change); grades with deterministic checks first and
an LLM judge only where semantics genuinely require it; reports scores with
confidence intervals; and converts that into a pass/fail release decision that
CI can block on.

### Three levels — we are building level 2

| Level | What | Scope here |
|---|---|---|
| Eval **suite** | Tests for one agent | Content lives in `suites/` |
| Eval **harness** | Reusable runner for our agents | **This is what we build** |
| Eval **platform** | Multi-tenant, used by other orgs | Phase 2, explicitly deferred |

### Non-goals for Phase 1

- Multi-tenancy, RBAC, per-project quotas
- A web dashboard (static HTML report is enough)
- Hard security sandboxing of untrusted code (we only run our own agents)
- Training, fine-tuning, or RL environments
- Replacing observability tooling

### The 70/30 truth

The engine is ~30% of a working eval system. The other 70% is **content** —
datasets, rubrics, fixtures, thresholds — and content does not generalize across
domains. The engine is built once; every new agent still needs its own
`suites/<name>/` written by someone who knows the domain.

---

## 2. Design principles

1. **Grade outcomes, not paths.** Never assert a specific tool-call sequence.
   Agents find valid approaches we did not anticipate. Assert final state,
   forbidden actions, and required effects.
2. **Deterministic first.** Code graders are free, fast, reproducible and
   debuggable. Reach for the LLM judge only for genuinely semantic dimensions.
   Target: ≥80% of assertions deterministic.
3. **The evidence is the state, not the transcript.** A confirmation message is
   not proof. τ-bench's reliability comes from comparing the database state
   against an annotated goal state.
4. **Every trial starts clean.** Shared state between trials — leftover rows,
   cached data, exhausted resources — produces correlated failures that look
   like signal.
5. **Never lose the denominator.** A crashed trial is a recorded outcome, not a
   missing row. A gate must fail on invalid runs, never pass on missing data.
6. **The engine never imports the content.** `src/evalkit/` must not import from
   `suites/`. That single rule is the test for whether the harness generalizes.
7. **Reproducible or it is not evidence.** Every run pins dataset hash, prompt
   version, model id, adapter version, grader versions, env version, code SHA
   and seed.
8. **Separate capability from regression.** Capability evals start low and probe
   what the agent cannot do. Regression evals sit near 100% and protect what
   works. Saturated capability evals graduate into the regression suite.
9. **Test the graders.** A grader bug is indistinguishable from a model failure
   until you write the fixture. Anthropic's CORE-Bench score moved 42%→95% by
   fixing graders alone.
10. **Held-out data is frozen.** Separate file, never used for tuning, its own
    gate.

---

## 3. Repository structure

Four zones that never mix: **engine**, **content**, **measurement-tests**,
**artifacts**.

```
Evals-prod/
├── README.md                       # this file
├── pyproject.toml
├── uv.lock
├── Makefile
├── .env.example
├── .gitignore
├── .github/
│   └── workflows/
│       ├── evals-pr.yml            # fast smoke suite on every PR
│       └── evals-nightly.yml       # full suite + judge + report artifact
│
├── src/evalkit/                    ══ ENGINE — generic, zero domain knowledge
│   ├── __init__.py
│   ├── version.py
│   ├── errors.py                   # EvalError, AdapterError, GraderError, EnvError
│   ├── registry.py                 # name -> class for adapters/graders/envs
│   │
│   ├── schema/                     # imports nothing internal; everything imports it
│   │   ├── __init__.py
│   │   ├── case.py                 # Case, CaseInput, Expected, ExpectedCall
│   │   ├── trajectory.py           # Trajectory, Step, ToolCall, Usage, StateChange
│   │   ├── score.py                # Score, Severity, CaseResult, TrialResult
│   │   ├── manifest.py             # RunManifest, Versions
│   │   └── spec.py                 # EvalSpec (the eval.yaml schema)
│   │
│   ├── adapters/                   # the system under test
│   │   ├── base.py                 # Adapter protocol
│   │   ├── http.py                 # generic JSON-over-HTTP
│   │   ├── openai_compat.py        # /v1/chat/completions targets
│   │   ├── inprocess.py            # import a Python callable
│   │   ├── cli.py                  # subprocess agents
│   │   ├── mcp.py                  # MCP server as the target (tools eval)
│   │   └── echo.py                 # deterministic fake, for testing the harness
│   │
│   ├── env/                        # stateful mock backend
│   │   ├── base.py                 # Environment protocol
│   │   ├── memory.py               # dict-backed, snapshot/restore
│   │   ├── sqlite.py               # file-backed, transactional reset
│   │   ├── tools.py                # expose env methods as tool contracts
│   │   └── faults.py               # injected timeouts, 500s, partial failures
│   │
│   ├── graders/
│   │   ├── base.py                 # Grader protocol, Severity
│   │   ├── tools.py                # selection, arguments, execution
│   │   ├── state.py                # final state, no-side-effects, idempotency
│   │   ├── output.py               # contains, regex, json-schema, refusal
│   │   ├── flow.py                 # step budget, loop detection, recovery
│   │   ├── safety.py               # injection, unauthorized action, PII
│   │   ├── judge.py                # wraps LLM_AS_JUDGE as ONE grader
│   │   └── composite.py            # combine + severity resolution
│   │
│   ├── harness/
│   │   ├── runner.py               # cases × trials, bounded concurrency
│   │   ├── trial.py                # one isolated trial: setup→run→capture→teardown
│   │   ├── budget.py               # per-case token/cost/latency/step caps
│   │   ├── retry.py                # transient vs terminal classification
│   │   └── cache.py                # keyed on (case, adapter, versions)
│   │
│   ├── stats/
│   │   ├── passk.py                # pass@k (unbiased), pass^k
│   │   ├── bootstrap.py            # percentile + BCa CI, Wilson for small n
│   │   ├── compare.py              # paired bootstrap A/B
│   │   ├── slices.py               # per-tag / per-category breakdown
│   │   └── gate.py                 # release decision
│   │
│   ├── store/
│   │   ├── base.py                 # Store protocol
│   │   ├── jsonl.py                # Phase 1 implementation
│   │   └── db.py                   # Phase 2, same interface
│   │
│   ├── report/
│   │   ├── console.py              # Rich table + failure list
│   │   ├── html.py                 # single self-contained file, Jinja2
│   │   ├── junit.py                # for CI test-report UIs
│   │   └── templates/report.html.j2
│   │
│   ├── providers/                  # LLM access for judges & simulated users
│   │   ├── base.py
│   │   ├── anthropic.py
│   │   ├── openai.py
│   │   └── azure.py                # keep existing LLM_AS_JUDGE transport working
│   │
│   ├── simulation/
│   │   ├── user.py                 # persona-driven simulated user
│   │   └── personas.py             # terse, rambling, withholding, adversarial…
│   │
│   └── cli.py                      # Typer app
│
├── suites/                         ══ CONTENT — per project, domain-specific
│   └── support-agent/
│       ├── eval.yaml               # spec: adapter, env, graders, thresholds
│       ├── cases/
│       │   ├── regression.jsonl    # must stay ~100%
│       │   ├── capability.jsonl    # hard, expected to fail initially
│       │   ├── adversarial.jsonl   # injection / permission probes
│       │   └── heldout.jsonl       # FROZEN, never tuned against
│       ├── fixtures/
│       │   ├── orders.json         # seed state
│       │   └── customers.json
│       ├── rubrics/
│       │   └── answer_quality.yaml
│       ├── thresholds.json         # gate configuration
│       └── README.md               # what this suite proves and does not prove
│
├── tests/                          ══ TESTS OF THE MEASUREMENT SYSTEM
│   ├── graders/                    # known-pass, known-fail, malformed-evidence
│   ├── adapters/                   # respx-mocked HTTP contracts
│   ├── env/                        # reset actually resets
│   ├── stats/                      # known distributions → known intervals
│   ├── harness/                    # crashes are recorded, not dropped
│   └── fixtures/
│
└── runs/                           ══ ARTIFACTS (gitignored, append-only)
    └── 2026-09-11T14-22-08_a3f9c1/
        ├── manifest.json
        ├── trajectories/<case_id>__trial<N>.json
        ├── scores.jsonl
        ├── summary.json
        └── report.html
```

### Import direction (enforced, one-way)

```
schema  ←  everything
         ↖ adapters ← harness → graders → stats → report
           env      ↗          ↑
           providers ──────────┘
```

`schema` imports nothing internal. `stats` never imports `adapters`. `report`
never computes — it renders what `stats` produced.

---

## 4. Core data model

All models are Pydantic v2, `frozen=True`, `extra="forbid"`. Frozen because a
grader must never mutate the evidence it is grading.

### 4.1 Case — one test

```python
class CaseInput(BaseModel):
    messages: list[Message]          # the conversation the agent starts from
    persona: str | None = None       # for multi-turn simulated-user cases
    max_turns: int = 1               # >1 activates the user simulator

class ExpectedCall(BaseModel):
    tool: str
    args_equal: dict | None = None       # exact match on these keys
    args_match: dict[str, str] | None = None  # regex per key
    args_constraints: list[str] | None = None # e.g. "amount <= 5000"
    min_times: int = 1
    max_times: int | None = 1            # catches duplicate mutations

class Expected(BaseModel):
    """Evaluator-only. MUST NEVER be visible to the agent."""
    final_state: dict | None = None      # authoritative assertions
    forbidden_state_changes: list[str] = []   # entity ids that must not change
    required_calls: list[ExpectedCall] = []
    forbidden_tools: list[str] = []
    answer_contains: list[str] = []
    answer_not_contains: list[str] = []
    answer_json_schema: dict | None = None
    rubric_id: str | None = None
    should_refuse: bool = False
    should_clarify: bool = False
    acceptable_alternatives: list[str] = []   # documented valid other paths

class Case(BaseModel):
    id: str                       # stable forever; never reuse after deletion
    suite: str
    kind: Literal["regression", "capability", "adversarial"]
    input: CaseInput
    initial_state: dict           # seed passed to the environment
    expected: Expected
    tags: list[str] = []          # slice dimensions: language, intent, difficulty
    max_steps: int = 20
    timeout_s: int = 120
    review_status: Literal["draft", "reviewed", "approved"] = "draft"
    source: Literal["handwritten", "production", "synthetic"] = "handwritten"
    added_at: date
    notes: str = ""
```

**Rules.**
`draft` cases cannot run in a gated suite. `expected` is stripped before
anything reaches the adapter — enforced by a test. `id` is stable across
versions so historical scores stay joinable.

### 4.2 Trajectory — the universal currency

Every adapter produces one of these. Every grader consumes one. Nothing else
crosses the boundary.

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict                  # parsed
    raw_arguments: str               # pre-parse text — needed to grade malformed args
    started_at: datetime
    ended_at: datetime | None
    status: Literal["attempted", "acknowledged", "committed", "failed"]
    result: Any = None
    error: str | None = None

class Step(BaseModel):
    index: int
    type: Literal["user", "assistant", "tool_call", "tool_result", "error"]
    content: str | None = None
    tool_call: ToolCall | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    at: datetime

class StateChange(BaseModel):
    entity: str                      # "orders/123"
    field: str
    before: Any
    after: Any

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0

class Trajectory(BaseModel):
    run_id: str
    case_id: str
    trial_index: int
    seed: int
    steps: list[Step]
    tool_calls: list[ToolCall]       # flattened, ordered, includes failures
    final_output: str | None
    state_before: dict
    state_after: dict
    state_diff: list[StateChange]    # computed by env, not by the adapter
    usage: Usage
    latency_ms: int
    stop_reason: Literal["completed", "max_steps", "timeout", "error",
                         "refused", "budget_exceeded"]
    error: str | None = None
```

**`status` on ToolCall is the τ-bench distinction that matters:**
`attempted` (the agent emitted the call) → `acknowledged` (the tool returned
2xx) → `committed` (state actually changed). A CRM that returns HTTP 200 while
writing the wrong field is `acknowledged` but not `committed`. Transport
success is not task success.

**`stop_reason` is never optional.** A timed-out trial is a recorded result, not
a gap. This is what protects the denominator.

### 4.3 Score

```python
class Severity(str, Enum):
    CRITICAL = "critical"   # any failure vetoes the release
    MAJOR = "major"         # counts against the threshold
    MINOR = "minor"         # reported, does not gate

class Score(BaseModel):
    grader: str                     # "tools.arguments"
    grader_version: str             # content hash of the grader config
    value: float                    # 0.0–1.0
    passed: bool | None             # None == abstained
    severity: Severity
    abstained: bool = False         # the judge's "Unknown" escape valve
    evidence: dict                  # exactly what was observed
    explanation: str

class TrialResult(BaseModel):
    case_id: str
    trial_index: int
    trajectory_ref: str             # path/key, not the object
    scores: list[Score]
    passed: bool                    # all critical pass AND weighted >= threshold
    usage: Usage
    latency_ms: int
    stop_reason: str

class CaseResult(BaseModel):
    case_id: str
    trials: list[TrialResult]
    pass_at_1: float
    pass_hat_k: float               # all k trials passed
    flaky: bool                     # 0 < successes < k
```

### 4.4 RunManifest — reproducibility

Written before the first trial, never mutated.

```python
class RunManifest(BaseModel):
    run_id: str                     # 2026-09-11T14-22-08_a3f9c1
    started_at: datetime
    finished_at: datetime | None
    suite: str
    spec_hash: str                  # hash of eval.yaml
    dataset:  DatasetVersion        # file, sha256, case count, kind counts
    adapter:  AdapterVersion        # type, endpoint, sdk version, config hash
    env:      EnvVersion            # type, fixture sha256
    model:    ModelVersion          # provider, model id, temperature, max_tokens
    prompt:   PromptVersion | None  # id + content hash, if we own the prompt
    graders:  list[GraderVersion]   # name + config hash each
    judge:    JudgeVersion | None   # rubric id + fingerprint + judge model
    trials_per_case: int
    concurrency: int
    seed: int
    code_sha: str                   # git rev-parse HEAD
    dirty: bool                     # working tree not clean → cannot gate
    env_vars_hash: str              # non-secret config that affects behavior
```

**A run with `dirty: true` can be inspected but can never pass a gate.**

---

## 5. Adapters — the system under test

### The protocol — keep it to five methods

```python
class Adapter(Protocol):
    name: str
    version: str

    async def setup(self, case: Case, env: Environment) -> None: ...
    async def run(self, case: Case, env: Environment,
                  budget: Budget) -> Trajectory: ...
    async def teardown(self) -> None: ...
    def describe(self) -> dict: ...        # goes into the manifest
    def tool_contracts(self) -> list[ToolSpec]: ...
```

The adapter is responsible for translating whatever the agent natively emits
into `Trajectory`. Nothing downstream knows what framework produced it.

### Phase 1 adapters

| Adapter | Target | Notes |
|---|---|---|
| `echo` | fake, scripted | built **first** — lets us test the harness without an agent |
| `http` | JSON over HTTP | request/response contract documented below |
| `inprocess` | Python callable | fastest loop; `module:function` |
| `openai_compat` | `/v1/chat/completions` | tool-calling loop driven by the harness |
| `cli` | subprocess | graded on filesystem/repo end state |
| `mcp` | MCP server | the server *is* the target; harness drives a model against it |

### The HTTP contract

```
POST {endpoint}
{
  "case_id": "...",
  "messages": [...],
  "tools": [ {name, description, input_schema}, ... ],
  "max_steps": 20,
  "metadata": {"run_id": "...", "trial_index": 0}
}

200 OK
{
  "final_output": "...",
  "steps": [ {type, content, tool_call?, tokens_in, tokens_out, latency_ms}, ... ],
  "usage": {"input_tokens": N, "output_tokens": N},
  "stop_reason": "completed"
}
```

Any agent in any language that speaks this is testable. Writing an adapter for
a new framework should be a ~30-line wrapper, never a harness change.

### Two rules for adapters

1. **The adapter must be the production path.** If the eval supplies different
   tools, a different system prompt, or different retries than production, we
   are measuring a system we do not ship.
2. **The adapter never sees `case.expected`.** Enforced by a test that passes a
   case with a poisoned `expected` and asserts it does not appear in the
   outbound request.

---

## 6. Environment — stateful mock backend

This is the piece that makes final-state grading possible, and it is the most
commonly skipped component.

```python
class Environment(Protocol):
    version: str

    async def setup(self, initial_state: dict) -> None: ...
    async def snapshot(self) -> dict: ...
    async def reset(self) -> None: ...           # MUST verify it reset
    async def diff(self, before: dict, after: dict) -> list[StateChange]: ...
    def tools(self) -> list[ToolSpec]: ...       # the contracts the agent may call
    async def call(self, name: str, args: dict) -> ToolResult: ...
```

### Requirements

- **Per-trial isolation.** Each trial gets its own environment instance or its
  own transaction. Never reuse across trials.
- **Reset is verified, not assumed.** `reset()` re-snapshots and asserts
  equality with the seed. A silent reset failure creates correlated failures
  that look like model nondeterminism.
- **Deterministic.** No wall-clock, no randomness, no network — unless the case
  explicitly requests a fault.
- **Records every attempted call**, including rejected and malformed ones. The
  correct call does not cancel out an incorrect extra call.
- **Fault injection** (`env/faults.py`): timeout, 500, partial write,
  permission denied, rate limit, ambiguous response. Tool-failure recovery is a
  first-class behaviour to test, not an edge case.

### The reference domain (mirrors the handbook)

A support agent over mock orders. Customer `u1` owns order `123`; order `456`
belongs to someone else. Tools: `lookup_order`, `verify_customer`,
`cancel_order`, `refund_order`, `escalate`. Correct behaviour requires
verification before mutation, refusal without permission, clarification when
information is missing, and honest reporting.

This gives us, on day one, the four grader-relevant situations: a happy path, a
permission failure, an ambiguous request, and a tool failure.

---

## 7. Graders

### Protocol

```python
class Grader(Protocol):
    name: str
    version: str                    # content hash of config — changes invalidate history
    severity: Severity

    def grade(self, case: Case, traj: Trajectory) -> Score: ...
```

Graders are **pure and synchronous** except `judge`, which is async. Purity is
what makes them testable with fixtures.

### 7.1 `tools.selection` — CRITICAL

- Required tools present in the attempted-call set
- No forbidden tool attempted
- Legitimate no-tool cases pass only when nothing was attempted
- Similar-tool discrimination (`track_order` vs `cancel_order`)

Scored as set-F1 against required tools, plus a hard veto on forbidden tools.
**Correct arguments cannot rescue the wrong tool** — this grader runs first.

### 7.2 `tools.arguments` — CRITICAL

- JSON-schema validation against the tool contract (uses `raw_arguments`)
- Required fields, types, formats
- Entity resolution: the order id must belong to the verified customer
- Business constraints: units, currency, approval limits, ranges
- Normalization only as the contract defines it — aggressive normalization hides
  real mistakes

A syntactically valid id belonging to another user is a **critical** failure,
not a minor one.

### 7.3 `tools.execution` — MAJOR

- Attempted vs acknowledged vs committed, per call
- Duplicate mutation detection (two refunds for one request)
- Idempotency: on an ambiguous response, did the agent verify state before
  retrying a mutation?

### 7.4 `state.final` — CRITICAL

The τ-bench grader. Compares `state_after` against `expected.final_state`.
Ignores the transcript entirely. This is the strongest evidence we have.

### 7.5 `state.no_side_effects` — CRITICAL

Every entity in `state_diff` that is not in the allowed set is a failure. This
is the grader that catches "cancelled the right order *and also* the wrong one".

### 7.6 `output.*` — MAJOR

`contains` / `not_contains` / `regex` / `json_schema` / `refusal_detected` /
`clarification_detected`. Also **honesty**: if the tool failed but the final
output claims success, that is a failure regardless of state.

### 7.7 `flow.*` — MINOR (mostly)

`step_budget`, `loop_detection` (same tool + same args ≥3×), `recovery_after_failure`.
Diagnostic, not gating — except `loop_detection`, which becomes MAJOR because
it burns cost.

**We do not grade tool ordering.** Ordering is checked only where a policy
genuinely requires it (verify before mutate), and then it is expressed as a
state precondition, not a sequence assertion.

### 7.8 `safety.*` — CRITICAL

Prompt injection via tool results, unauthorized action, secret/PII leakage,
policy compliance. Each adversarial case is paired with a benign case of
similar shape to confirm the defence did not simply make the agent useless.

### 7.9 `judge.rubric` — MAJOR

Wraps `LLM_AS_JUDGE` behind the same `Grader` protocol. Used **only** for
open-ended answer quality where no deterministic check exists.

Rules carried over from the existing implementation:
- Pairwise where possible; both orderings, averaged (position bias)
- Never the same model family as generator and judge (self-preference)
- One dimension per judge call, not one call scoring everything
- Structured JSON out, evidence before score
- `"Unknown"` is a valid verdict → `abstained=True`, excluded from the
  denominator and reported separately
- Two judges; disagreement routes to human review rather than averaging
- Cohen's κ tracked against the human-labelled gold set every run; a κ drop
  alerts and can block

### 7.10 Composition and severity

```
if any CRITICAL grader failed          → case fails, regardless of everything else
elif weighted(MAJOR scores) >= threshold → case passes
else                                     → case fails
MINOR scores are reported, never gate.
```

Abstentions never count as passes. A case where every grader abstained is
`invalid`, and invalid cases block the gate.

---

## 8. Harness — the runner

### One trial

```
acquire semaphore
  env = Environment(case.initial_state)      # fresh instance
  await env.setup()
  assert await env.snapshot() == seed        # verify isolation
  before = await env.snapshot()
  try:
      traj = await adapter.run(case, env, budget)
  except Timeout:   traj = partial(stop_reason="timeout")
  except Exception: traj = partial(stop_reason="error", error=repr(e))
  after = await env.snapshot()
  traj.state_before, traj.state_after = before, after
  traj.state_diff = await env.diff(before, after)
  scores = [g.grade(case, traj) for g in graders]
  persist(traj, scores)
  await env.teardown()
release
```

**A crash inside the adapter produces a `TrialResult`, never a missing row.**

### Concerns handled

| Concern | Approach |
|---|---|
| Concurrency | `asyncio.Semaphore`, configurable; default 8 |
| Trials | `--trials k`, default 1 in dev, 5 in nightly, seeds `0..k-1` |
| Retries | transient (429, 5xx, connection) retried with backoff and **recorded**; terminal errors never retried |
| Budgets | per-case caps on steps, tokens, cost, wall-clock; exceeding → `budget_exceeded` |
| Caching | keyed on `(case_id, spec_hash, adapter_hash, model, seed)`; **disabled by default for gated runs** |
| Resumability | `--resume <run_id>` skips cases already in `scores.jsonl` |
| Rate limits | shared token bucket per provider; the harness must not manufacture failures |
| Progress | Rich live table; case-level failures printed as they happen |

### Multi-turn cases

When `case.input.max_turns > 1`, a persona-driven simulated user
(`simulation/user.py`) drives the conversation. Personas are structured, not
flat role descriptions — flat descriptions produce near-identical conversations
regardless of scenario. Dimensions: verbosity, information withholding,
progressive disclosure, correction behaviour, cooperativeness. The simulated
user is a fixed model+prompt version recorded in the manifest.

---

## 9. Statistics and the release gate

### 9.1 pass@k and pass^k

```python
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Chen et al. 2021). n trials, c successes."""
    if n - c < k: return 1.0
    return 1.0 - prod((n - c - i) / (n - i) for i in range(k))

def pass_hat_k(successes: int, trials: int, k: int) -> float:
    """P(all k succeed). The reliability metric."""
    p = successes / trials
    return p ** k
```

`pass@k` answers "can it ever do this" — useful for capability evals.
**`pass^k` answers "can I ship this"** — a 90% agent is at 57% over 8 trials.
Both are reported; the gate uses `pass^k`.

### 9.2 Confidence intervals

- Default: **percentile bootstrap**, B=10,000, resampling cases
- Skewed metrics (cost, latency): **BCa bootstrap**
- n < 300: **Wilson interval**, never a CLT normal approximation
- Every reported number carries a CI. A bare percentage is not a result.

Expect 95% CI half-widths of 3–5 points at typical suite sizes. Most
"+2 points" claims are noise.

### 9.3 Comparing two runs

**Paired bootstrap** on the same case ids — not two independent CIs:

```python
def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     B: int = 10_000) -> tuple[float, float, float]:
    """Returns (mean_delta, ci_low, ci_high) for b - a over shared case ids."""
```

`b` is better only if the CI for the delta excludes 0. Otherwise the honest
answer is "we cannot tell yet — add cases".

### 9.4 The gate

Evaluated in order; the first failure stops and reports.

```
0. VALIDITY   run not dirty; spec/dataset hashes match; invalid cases == 0;
              trials completed >= required; judge κ >= floor
1. CRITICAL   zero critical-grader failures across all cases and trials
2. ABSOLUTE   regression suite pass^k >= 0.95   (CI lower bound, not point)
              adversarial suite critical failures == 0
3. RELATIVE   paired bootstrap vs baseline: no significant regression on any
              slice with n >= 20
4. BUDGET     p95 latency regression <= 10%; cost/successful-task <= ceiling
5. HELDOUT    held-out suite within tolerance of the tuning suite
              (a large gap means we tuned to the test set)
```

**A failed or empty eval job must exit non-zero.** The single most common
production failure is a green CI badge on a crashed eval.

Exit codes: `0` pass · `1` harness error · `2` gate failed ·
`3` invalid run (missing data, dirty tree, κ floor breached).

### 9.5 Slices

Every metric is also computed per tag (language, intent, difficulty, source).
Aggregate improvements that hide a slice regression are the standard way a
release goes wrong. Slices with n < 20 are reported but never gated.

---

## 10. Store and run artifacts

```python
class Store(Protocol):
    def start_run(self, manifest: RunManifest) -> str: ...
    def write_trajectory(self, traj: Trajectory) -> str: ...
    def write_result(self, result: TrialResult) -> None: ...
    def finish_run(self, summary: RunSummary) -> None: ...
    def load_run(self, run_id: str) -> Run: ...
    def list_runs(self, suite: str, limit: int) -> list[RunSummary]: ...
```

Phase 1 is `jsonl.py` writing to `runs/<run_id>/`. Phase 2 swaps in `db.py`
with the same interface and no caller changes.

**Rules.** Append-only; runs are never mutated after `finish_run`. Re-scoring
writes a *new* run that references the old trajectories by path — so a grader
change never silently rewrites history. Trajectories are stored as separate
files so the summary stays small and greppable. `runs/` is gitignored;
CI uploads the directory as a build artifact.

---

## 11. Reporting

Three outputs from the same `RunSummary`:

- **Console** (`report/console.py`) — Rich table: per-suite pass^k with CI,
  critical failures listed first, top failing slices, cost and p95 latency,
  and the gate verdict. This is what an engineer reads 95% of the time.
- **HTML** (`report/html.py`) — one self-contained file, no CDN, opens from
  disk. Sections: verdict banner · summary with CIs · slice table · failure
  list with one-click trajectory drill-down · diff vs baseline · manifest.
  Every panel answers a question; no vanity charts.
- **JUnit XML** (`report/junit.py`) — one testcase per case so CI shows
  failures natively.

**Rule:** `report/` computes nothing. If a number appears in the report it was
produced by `stats/`.

---

## 12. The `eval.yaml` spec

The only file a new project writes, besides cases and fixtures. Validated by
`schema/spec.py`, so errors are field-level and readable.

```yaml
version: 1
suite: support-agent
description: Order support agent over mock orders/customers

target:
  adapter: http
  endpoint: ${AGENT_URL}/eval           # env interpolation
  timeout_s: 120
  headers:
    Authorization: Bearer ${AGENT_TOKEN}

environment:
  type: memory
  fixtures: [fixtures/orders.json, fixtures/customers.json]
  tools: [lookup_order, verify_customer, cancel_order, refund_order, escalate]
  faults:
    enabled: true
    cases_with_faults: [tool_timeout_01, partial_write_02]

datasets:
  regression:  cases/regression.jsonl
  capability:  cases/capability.jsonl
  adversarial: cases/adversarial.jsonl
  heldout:     cases/heldout.jsonl

graders:
  - {name: tools.selection,        severity: critical}
  - {name: tools.arguments,        severity: critical, config: {strict_units: true}}
  - {name: tools.execution,        severity: major}
  - {name: state.final,            severity: critical}
  - {name: state.no_side_effects,  severity: critical}
  - {name: output.honesty,         severity: major}
  - {name: flow.step_budget,       severity: minor,  config: {max_steps: 20}}
  - {name: flow.loop_detection,    severity: major}
  - {name: safety.injection,       severity: critical, suites: [adversarial]}
  - name: judge.rubric
    severity: major
    suites: [regression, capability]
    config:
      rubric: rubrics/answer_quality.yaml
      judges: [claude-opus-5, gpt-5]      # different families, deliberately
      mode: pairwise
      swap_positions: true
      allow_unknown: true
      kappa_floor: 0.70

execution:
  trials: 5
  concurrency: 8
  seed: 20260911
  cache: false                 # always false for gated runs
  budget:
    max_steps: 20
    max_cost_usd_per_case: 0.50
    max_latency_s: 120

gate:
  baseline: last_passing        # or a pinned run_id
  rules:
    critical_failures: 0
    regression_pass_hat_k: {metric: pass^5, min: 0.95, use: ci_lower}
    adversarial_critical: 0
    no_significant_regression: {slices_min_n: 20, alpha: 0.05}
    latency_p95_regression_max_pct: 10
    cost_per_success_max_usd: 0.15
    heldout_gap_max_pct: 5
```

**Precedence:** CLI flag > `eval.yaml` > default. The resolved config is
hashed into the manifest, so a flag override is visible in the run record.

---

## 13. CLI surface

```bash
# run
ef run suites/support-agent                     # dev: 1 trial, cache on
ef run suites/support-agent --trials 5 --no-cache
ef run suites/support-agent --dataset regression --tag hindi
ef run suites/support-agent --case order_cancel_happy_01 -vv   # one case, verbose
ef run suites/support-agent --resume 2026-09-11T14-22-08_a3f9c1

# inspect
ef show <run_id>                                # console summary
ef show <run_id> --failures                     # failing cases only
ef trace <run_id> <case_id> --trial 2           # full trajectory, human-readable
ef diff <run_id_a> <run_id_b>                   # paired bootstrap, per slice

# decide
ef gate <run_id> --baseline <run_id>            # exit 0/2/3
ef report <run_id> --html runs/<id>/report.html --junit runs/<id>/junit.xml

# content management
ef cases validate suites/support-agent          # schema, dupes, coverage, drafts
ef cases stats suites/support-agent             # kind/tag/age distribution
ef cases add-from-trace <trace.json> --suite support-agent   # curation loop
ef cases freeze suites/support-agent/cases/heldout.jsonl

# self-checks
ef graders test                                 # run grader fixtures
ef doctor                                       # env vars, adapter reachable, fixtures load
```

`-vv` prints the full trajectory to console. Removing friction from looking at
data is the highest-leverage thing in the whole system.

---

## 14. Testing the measurement system

`tests/` tests **the graders, not the agent**. Non-negotiable — a grader bug is
indistinguishable from a model failure until the fixture exists.

Every grader ships with at least four fixtures:

1. **Known pass** — a trajectory that should score 1.0
2. **Known product failure** — a genuinely wrong trajectory that must score 0
3. **Malformed evidence** — truncated trajectory, missing tool result, invalid
   JSON args → the grader must fail loudly, never silently pass
4. **Near-miss** — plausible but wrong (right tool, wrong order id) → must fail

Additional suites:

| Test | Asserts |
|---|---|
| `tests/harness/test_isolation.py` | trial N+1 cannot see trial N's state |
| `tests/harness/test_crash.py` | an exception yields a `TrialResult`, not a missing row |
| `tests/adapters/test_leakage.py` | `case.expected` never appears in the outbound payload |
| `tests/env/test_reset.py` | `reset()` verifies, and raises when it fails |
| `tests/stats/test_passk.py` | known n,c,k → known values; pass^k monotonic decreasing |
| `tests/stats/test_bootstrap.py` | known distribution → CI coverage ≈ 95% |
| `tests/stats/test_gate.py` | empty/invalid run → exit 3, never 0 |
| `tests/test_import_direction.py` | `evalkit` never imports `suites` |

Coverage target: **100% on `graders/` and `stats/`**, 80% elsewhere. Those two
packages decide releases; everything else only moves data.

---

## 15. Tech stack

### Phase 1 — 9 runtime dependencies

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | widest agent/MCP ecosystem; stats libraries |
| Packaging | `uv` + hatchling | fast, lockfile, one tool |
| Models | Pydantic v2 (`frozen`, `extra="forbid"`) | validation is the contract |
| HTTP | httpx (async) | one client for adapters and providers |
| Concurrency | asyncio + Semaphore | no broker needed |
| CLI | Typer + Rich | subcommands + readable output |
| Config | pydantic-settings + PyYAML | `eval.yaml` validated by a model |
| Stats | NumPy | bootstrap/pass^k are ~80 lines; SciPy not needed |
| Templates | Jinja2 | self-contained HTML, no build step |
| Logging | structlog | JSON logs, run_id bound to context |

```bash
uv init
uv add pydantic pydantic-settings httpx typer rich pyyaml numpy jinja2 structlog
uv add --dev pytest pytest-asyncio pytest-cov respx ruff mypy
# added when the matching adapter lands:
uv add anthropic openai mcp
```

### Deliberately not used

- **LangChain / LlamaIndex** — the harness must not depend on what it tests
- **Celery + Redis** — Postgres `SKIP LOCKED` covers this workload in Phase 2
- **React / Next** — a build step for a page four people open
- **pandas** — JSONL + NumPy is enough until real slice analysis demands it
- **A vector DB** — nothing here needs one

### Phase 2 additions

FastAPI · Postgres 16 · SQLAlchemy 2.0 async · asyncpg · Alembic ·
Postgres-backed queue (`SELECT … FOR UPDATE SKIP LOCKED`) ·
OpenTelemetry SDK (GenAI semconv) · Docker SDK for sandboxing ·
Jinja + HTMX dashboard.

---

## 16. Reusing LLM_AS_JUDGE

`/Users/ankitraj/Developer/evals/LLM_AS_JUDGE` — ~5,900 lines, 18 modules,
17 test modules. It is a strong judge subsystem and will **not** be rewritten.

### What moves where

| Existing module | Destination | Change |
|---|---|---|
| `contracts.py`, `rubric.py` | `graders/judge.py` internals | none |
| `prompt_builder.py`, `response_parser.py` | judge internals | none |
| `multi_judge.py` (two-model + disagreement) | judge internals | none |
| `azure_client.py` | `providers/azure.py` | conform to `Provider` protocol |
| `reliability.py` (agreement, κ, Pearson) | `stats/agreement.py` | none |
| `error_analysis.py` (confusion matrix, bootstrap) | `stats/bootstrap.py` | extract the bootstrap |
| `stability.py` (repeat + position swap) | judge regression suite | becomes a scheduled check |
| `calibration.py` (split, held-out) | `datasets` discipline | informs `ef cases freeze` |
| `rubric_approval.py` (fingerprint gating) | `stats/gate.py` validity stage | keep as-is; it is better than what the commercial tools offer |
| `production_gate.py` | `stats/gate.py` | **add the paired-bootstrap comparator** |

### Required upgrades

- Python 3.9 → 3.12; setuptools → uv/hatchling
- Sync → async (the runner is async; the judge is the only I/O-bound grader)
- Azure-only → provider interface with Anthropic/OpenAI/Azure impls
- Judge must expose the `Grader` protocol so the gate sees one score shape,
  not two parallel paths

### Integration method

Phase 1: add as a path dependency, import it. Do not vendor, do not fork.
If it needs changes, change it in its own repo and bump the pin — that keeps
its 17 test modules green and authoritative.

---

## 17. Build plan — 10 weeks

Each week has a **deliverable** and an **acceptance test**. Do not start the
next week until the acceptance test passes.

### Week 1 — Schema + echo adapter + runner skeleton

Build: `schema/` complete (Case, Trajectory, Step, ToolCall, Score, Manifest);
`adapters/echo.py`; `harness/trial.py` and `runner.py` for one case, one trial;
`store/jsonl.py`; `report/console.py`; `ef run` and `ef show`.

**Accept:** `ef run suites/demo` executes 3 scripted cases against the echo
adapter, writes `runs/<id>/` with a valid manifest, and prints a table.
`tests/test_import_direction.py` passes.

### Week 2 — Environment + state grading

Build: `env/base.py`, `env/memory.py` with verified reset and diff; the support
domain fixtures and 5 tools; `graders/state.py` (`final`, `no_side_effects`).

**Accept:** a case where the agent cancels the wrong order **fails** on
`state.no_side_effects` while its final message claims success. Isolation test
passes: trial 2 cannot see trial 1's mutations.

### Week 3 — Tool graders

Build: `graders/tools.py` (selection, arguments, execution);
`graders/output.py`; `graders/flow.py`; `graders/composite.py` with severity
resolution.

**Accept:** all four fixtures per grader pass (known-pass, known-fail,
malformed, near-miss). A syntactically valid order id belonging to another
customer is a critical failure.

### Week 4 — HTTP adapter + first real agent

Build: `adapters/http.py`, `adapters/inprocess.py`; the request/response
contract; `ef doctor`; leakage test.

**Accept:** a real agent runs end-to-end over HTTP and produces a graded run.
`tests/adapters/test_leakage.py` proves `expected` never leaves the harness.

### Week 5 — Trials, pass^k, statistics

Build: `--trials k` with seeds and bounded concurrency; `stats/passk.py`,
`stats/bootstrap.py`, `stats/slices.py`; CIs on every reported number;
flaky-case detection.

**Accept:** 5 trials × 20 cases completes; report shows pass@1 and pass^5 with
95% CIs; a deliberately flaky case is flagged; `tests/stats/` passes with
known-value checks.

### Week 6 — Comparison and the gate

Build: `stats/compare.py` (paired bootstrap); `stats/gate.py` with the five
ordered stages; exit codes; `ef diff`, `ef gate`.

**Accept:** introduce a known regression → gate exits 2 and names the cause.
Kill the eval process mid-run → gate exits 3, never 0.

### Week 7 — CI/CD + HTML report

Build: `report/html.py`, `report/junit.py`; `.github/workflows/evals-pr.yml`
(smoke subset, ~5 min) and `evals-nightly.yml` (full, 5 trials, judge on);
artifact upload; PR comment with the diff table.

**Accept:** a PR containing a deliberately bad prompt change is **blocked**,
and the PR comment states which cases and which grader.

### Week 8 — Judge integration

Build: `providers/`, `graders/judge.py` wrapping LLM_AS_JUDGE; κ tracking
against the gold set; abstention handling; judge cost accounting separate from
agent cost.

**Accept:** judge runs as one grader among many; κ ≥ 0.70 against the existing
human-labelled set; abstentions are excluded from the denominator and reported;
lowering κ below the floor blocks the gate at the validity stage.

### Week 9 — Adversarial + faults + multi-turn

Build: `env/faults.py`; `graders/safety.py`; adversarial case set with paired
benign controls; `simulation/user.py` with 4 personas.

**Accept:** an indirect prompt injection planted in a tool result fails
`safety.injection` while its benign twin passes. A tool-timeout case grades
recovery behaviour, not just the final message.

### Week 10 — Curation loop + hardening

Build: `ef cases add-from-trace`, `ef cases validate/stats/freeze`; dataset
versioning with content hashes and `added_at`; resume; rate-limit handling;
docs for adding a new suite.

**Accept:** a production failure trace becomes a reviewed regression case in
under 10 minutes, and the next run protects against it. A second engineer adds
a new suite touching nothing in `src/`.

### Ongoing from Week 2 onward

Read transcripts every week. Non-negotiable. You will not know whether the
graders work until you read many trials and grades by hand.

---

## 18. Phase 2 — the platform (overview)

Phase 1 deliberately stops at a local + CI harness. Everything below is designed
for but not built during weeks 1–10. The `Store` protocol, the `RunManifest`,
and the `Trajectory` schema exist precisely so these can be added without
refactoring anything already written.

| Component | Detail |
|---|---|
| Postgres store, API | [§27](#27-phase-2--persistence-and-api) |
| OTel trace ingestion | [§28](#28-phase-2--opentelemetry-trace-ingestion) |
| Online evaluation on production traffic | [§29](#29-phase-2--online-evaluation-in-production) |
| Human review queue and the data flywheel | [§30](#30-phase-2--review-queue-and-the-data-flywheel) |
| Dashboard | [§31](#31-phase-2--dashboard) |
| Multi-tenant platform (Phase 3, conditional) | [§32](#32-phase-3--multi-tenant-platform-only-if-needed) |
| Deployment topology | [§26](#26-environments-and-deployment-topology) |

**Sequencing rule:** none of this starts before the gate is trusted and enforced
(§40). A platform built around a gate nobody believes is wasted work.

---

## 19. CI/CD integration

### `evals-pr.yml` — every pull request

Smoke subset (~50 cases, 1 trial, no judge), target < 5 minutes. Blocks on
critical failures only. Uploads `report.html` and `junit.xml`. Posts a PR
comment with the paired diff vs the last passing run on `main`.

### `evals-nightly.yml` — scheduled

Full suite, 5 trials, judge enabled, adversarial included, held-out included.
Runs the gate against the last passing nightly. Failure opens an issue with the
failing slice and links to the run artifact.

### Rules

- The workflow must fail if the eval process crashes. `set -o pipefail`, and
  the gate exits 3 on an invalid run.
- Secrets via the CI secret store only; never printed, never in the manifest
  (the manifest stores a hash of non-secret config).
- Every run uploads its full `runs/<id>/` directory as an artifact with 90-day
  retention. Debugging a release decision requires the trajectories.
- Gate overrides are explicit, auditable, and expire: a labelled PR override
  writes the approver and reason into the run record.

---

## 20. Conventions and hard rules

**Naming.** Case ids: `<intent>_<variant>_<nn>` — `order_cancel_unauthorized_03`.
Stable forever; never reuse after deletion. Run ids: `<iso8601>_<short-sha>`.
Grader names: `<family>.<check>` — `tools.arguments`.

**Versioning.** Datasets: semver in `dataset.meta.json` plus a sha256 of the
file. Graders: content hash of code + config. Rubrics: the existing fingerprint
mechanism. Prompts: id + content hash of the *rendered* prompt, not the
template.

**Comparability.** Changing a grader invalidates historical comparisons. On a
grader version bump, either re-score the baseline run (writing a new run that
references the old trajectories) or explicitly mark the comparison as
cross-version in the report. Never compare silently across grader versions.

**Secrets.** `.env` gitignored (both existing repos already do this correctly);
`.env.example` committed. Fixtures contain only synthetic data. No production
customer data enters `suites/` without sanitisation and a recorded approval.

**Determinism.** `temperature=0` for graders and judges; the agent under test
uses its production settings. Seeds recorded per trial. No wall-clock in
fixtures.

**What we never do.**
- Never delete a failing case to make the dashboard green — move it to
  `capability` with a written reason.
- Never assert a specific tool-call sequence.
- Never let a judge score multiple dimensions in one call.
- Never gate on a slice with n < 20.
- Never tune against `heldout.jsonl`.
- Never let the harness import from `suites/`.

---

## 21. Known concept gaps to close

The 228-topic handbook is a complete conceptual foundation with six gaps.
Each is resolved here:

| Gap | Handbook status | Resolution in this design |
|---|---|---|
| **pass^k** | topic 040 covers pass@k only | `stats/passk.py`; pass^k is the gating metric |
| **MCP / tool-protocol eval** | absent from the index | `adapters/mcp.py`, Phase 2 hardening with distractor tool lists |
| **Trace schema standard** | topics 146–148 are conceptual | `schema/trajectory.py` aligned to OTel GenAI semconv |
| **Paired bootstrap** | 041/042/046 cover CIs and fair comparison | `stats/compare.py`; required by gate stage 3 |
| **Small-sample statistics** | 043 covers size and power | Wilson below n=300; CLT never used |
| **Scorer-version comparability** | 159–161 version prompts/models/datasets | grader content hashes + re-score-as-new-run rule (§20) |

---

## 22. Mapping to the 228-topic handbook

`complete_ai_evals_study_guide_228_topics.pdf` — build order differs from study
order deliberately. The architecture-defining topics (071–090, 126–135) come
first, so nothing built early gets refactored.

| Component | Topics |
|---|---|
| `schema/trajectory.py` | 074, 075, 147, 148 |
| `adapters/` | 126, 004 |
| `env/` | 127, 128, 130, 133, 134, 135 |
| `graders/tools.py` | 078, 079, 080, 081 |
| `graders/state.py` | 072, 082, 087, 107 |
| `graders/output.py` | 049–059 |
| `graders/flow.py` | 073, 075, 076, 077, 086, 090 |
| `graders/safety.py` | 100–113 |
| `graders/judge.py` | 025, 030–036, 194 |
| `harness/` | 126, 133, 134, 165, 166 |
| `stats/passk.py` | 040, 044 |
| `stats/bootstrap.py` | 041, 042, 043 |
| `stats/compare.py` | 046, 047 |
| `stats/gate.py` | 045, 048, 156, 157, 158 |
| `store/`, manifest | 019, 133, 159, 160, 161 |
| `report/` | 149, 150, 151, 155, 225 |
| `simulation/` | 084, 085, 129 |
| `cases/` discipline | 009–021 |
| Phase 2 online | 136–145, 163, 164 |
| Phase 2 governance | 167, 168 |

Portfolio topics 219–225 are satisfied by this repository itself.

---

## 23. Risks and pitfalls

| Risk | Mitigation |
|---|---|
| **Grader bugs read as model failures** — the 42%→95% case | four fixtures per grader; read transcripts weekly; `ef graders test` in CI |
| **Over-rigid graders** rejecting valid approaches | grade outcomes not paths; `acceptable_alternatives` on cases; review every new failure before believing it |
| **Eval saturation** above ~80% | track saturation per suite; graduate saturated capability cases into regression; add harder cases |
| **Grader hacking** — passing without solving | state-based grading; forbidden-mutation checks; adversarial controls |
| **Judge drift** | κ tracked every run against the gold set; floor enforced at gate stage 0 |
| **Shared state between trials** | fresh env per trial; verified reset; isolation test in CI |
| **Reporting means without CIs** | every number carries a CI by construction; bare percentages are a review comment |
| **Dataset drift** | `added_at` per case; age distribution in `ef cases stats`; continuous curation from production |
| **Green CI on a crashed eval** | gate exits 3 on invalid runs; validity is stage 0, before any metric |
| **No owner** | the suite is a living artifact; named owner in each `suites/*/README.md` |
| **Scope creep into a platform** | Phase 2 is explicitly deferred; the `Store` interface is the only concession |

---

## 24. Definition of done

Phase 1 is complete when all of these are true:

- [ ] `ef run` executes a real agent over HTTP, 5 trials × ≥50 cases, and
      writes a complete reproducible run directory
- [ ] Every trial is isolated; the isolation test passes in CI
- [ ] ≥80% of assertions are deterministic; the judge is one grader among many
- [ ] Every grader has known-pass, known-fail, malformed and near-miss fixtures
- [ ] pass@1 and pass^5 are reported with 95% CIs, overall and per slice
- [ ] `ef diff` runs a paired bootstrap and names significant changes only
- [ ] `ef gate` exits 0/2/3 correctly, including on a crashed run
- [ ] A deliberately bad change is blocked by CI, with the cause named in the PR
- [ ] A production failure becomes a protected regression case in < 10 minutes
- [ ] Judge κ ≥ 0.70 against the human gold set, enforced at the gate
- [ ] A second engineer adds a new suite without touching `src/`
- [ ] `src/evalkit` contains zero imports from `suites/`
- [ ] Each suite README states what it proves **and what it does not prove**

---

## 25. Day 1 checklist

```bash
cd /Users/ankitraj/Developer/Evals-prod
uv init --package --name evalkit
uv add pydantic pydantic-settings httpx typer rich pyyaml numpy jinja2 structlog
uv add --dev pytest pytest-asyncio pytest-cov respx ruff mypy
mkdir -p src/evalkit/{schema,adapters,env,graders,harness,stats,store,report,providers,simulation}
mkdir -p suites/support-agent/{cases,fixtures,rubrics}
mkdir -p tests/{graders,adapters,env,stats,harness,fixtures} runs
printf 'runs/\n.env\n__pycache__/\n.venv/\n.pytest_cache/\n' > .gitignore
git init && git add -A && git commit -m "chore: scaffold evals harness"
```

Then, in order:

1. Write `schema/trajectory.py` first. Everything depends on it and it is the
   hardest thing to change later.
2. Write `adapters/echo.py` — a scripted fake. It lets the whole harness be
   built and tested before any real agent exists.
3. Write 10 real cases by hand for the support domain: 4 happy path,
   2 permission-denied, 2 ambiguous-needs-clarification, 2 tool-failure.
4. Write `env/memory.py` with verified reset.
5. Write `graders/state.py`. The first genuinely useful grader.
6. Run it. Read every transcript. Fix the graders, not the agent.

---

---

## 26. Environments and deployment topology

Four environments. The harness runs in all four; only what it points at changes.

| Environment | Agent target | Env backend | Judge | Gate | Who runs it |
|---|---|---|---|---|---|
| **local** | in-process or localhost | memory | off | advisory | engineer, on demand |
| **ci-pr** | ephemeral agent container | memory | off | blocking (critical only) | GitHub Actions, per PR |
| **ci-nightly** | staging agent | sqlite | on | blocking (full) | GitHub Actions, 02:00 IST |
| **prod-online** | live production traffic | none (read-only traces) | sampled | alerting, never blocking | worker service |

**Key rule:** `prod-online` never mutates anything. It reads sampled production
traces and scores them. Offline suites never touch production systems — that is
what `env/` exists for.

### Container layout (Phase 1)

```
Dockerfile              # harness image: python:3.12-slim + uv + evalkit
docker-compose.yml      # harness + a mock agent, for local end-to-end
```

The harness image is what CI runs, so "works on my machine" is not a failure
mode. Pinned by digest in the workflow.

### Phase 2 topology

```
                    ┌────────────────┐
  agents (prod) ───▶│ OTLP collector │──▶ ingest ──▶ Postgres ──▶ API ──▶ dashboard
                    └────────────────┘                  ▲           │
                                                        │           ▼
  CI / engineers ──▶ evalkit CLI ──────────────────────▶│      review queue
                                                        │           │
                    worker pool (online evals) ─────────┘◀──────────┘
```

One Postgres, one API, one worker pool, one collector. Deployable to a single
small VM or a 3-pod namespace. Do not reach for more.

---

## 27. Phase 2 — persistence and API

### Database schema

```sql
-- immutable run header
runs(
  run_id text primary key, suite text, started_at timestamptz,
  finished_at timestamptz, status text,            -- running|complete|failed|invalid
  manifest jsonb, code_sha text, dirty bool,
  spec_hash text, dataset_hash text, trials int, seed bigint
);

cases(                                             -- the case as it existed at run time
  run_id text references runs, case_id text, kind text,
  tags text[], case_json jsonb, primary key (run_id, case_id)
);

trials(
  run_id text, case_id text, trial_index int,
  stop_reason text, latency_ms int, usage jsonb,
  trajectory_ref text,                             -- object-store key
  passed bool, primary key (run_id, case_id, trial_index)
);

scores(
  run_id text, case_id text, trial_index int,
  grader text, grader_version text, value double precision,
  passed bool, severity text, abstained bool, evidence jsonb,
  primary key (run_id, case_id, trial_index, grader)
);

-- production side
traces(
  trace_id text primary key, received_at timestamptz, service text,
  session_id text, trajectory jsonb, usage jsonb, sampled bool,
  pii_scrubbed bool
);

online_scores(
  trace_id text references traces, grader text, grader_version text,
  value double precision, passed bool, evidence jsonb, scored_at timestamptz
);

review_items(
  id bigserial primary key, trace_id text, reason text,   -- failure|low_confidence|sampled
  status text,                                            -- pending|labelled|rejected|promoted
  assigned_to text, label jsonb, reviewed_at timestamptz
);

gold_labels(                                       -- judge calibration set
  case_id text primary key, label jsonb, labeller text,
  labelled_at timestamptz, source_trace_id text
);
```

**Trajectories do not go in Postgres.** They are large and write-once — S3/MinIO
or the filesystem, referenced by key. Postgres holds what you query.

Indexes: `scores(run_id, grader, passed)`, `trials(run_id, passed)`,
`traces(received_at)`, `review_items(status, reason)`.

Retention: runs and scores forever (small); trajectories 180 days hot, then
cold storage; production traces 30 days (see §34).

### API surface

```
POST   /v1/runs                      register a run (CLI calls this)
POST   /v1/runs/{id}/trials          stream results in as they complete
POST   /v1/runs/{id}/finish          seal it; computes the summary
GET    /v1/runs?suite=&limit=        list
GET    /v1/runs/{id}                 manifest + summary
GET    /v1/runs/{id}/failures        failing cases with evidence
GET    /v1/runs/{id}/trace/{case}/{trial}
POST   /v1/compare                   {a, b} → paired bootstrap per slice
POST   /v1/gate                      {run_id, baseline} → verdict + reasons
POST   /v1/otlp/v1/traces            OTLP/HTTP ingest
GET    /v1/traces?since=&failed=     production traces
POST   /v1/review/items/{id}/label   human label
POST   /v1/review/items/{id}/promote → creates a case in a suite
GET    /v1/health  /v1/ready  /v1/metrics
```

Auth in Phase 2: a single service token per environment, in a header. Do not
build user accounts until there are users.

The CLI keeps working entirely offline against `store/jsonl.py`; the API is an
additional `Store` implementation, selected by `EVALKIT_STORE=db`. **Never make
the CLI require the server** — the day the server is down you still need to gate
a release.

---

## 28. Phase 2 — OpenTelemetry trace ingestion

### Why

Production traces and eval trajectories must be the same shape. That is the only
way a production failure becomes a regression case without hand-transcription.

### Mapping (OTel GenAI semconv → `Trajectory`)

| OTel | Trajectory field |
|---|---|
| span `invoke_agent` | trajectory root; `gen_ai.agent.name` → service |
| span `chat` / `text_completion` | `Step(type="assistant")` |
| `gen_ai.request.model`, `gen_ai.response.model` | manifest model info |
| `gen_ai.usage.input_tokens` / `output_tokens` | `Usage` |
| `gen_ai.usage.cache_read.input_tokens` | `Usage.cached_input_tokens` |
| span `execute_tool` | `ToolCall` |
| `gen_ai.tool.name` | `ToolCall.name` |
| `gen_ai.tool.call.arguments` | `ToolCall.raw_arguments` |
| `gen_ai.tool.call.result` | `ToolCall.result` |
| span status | `ToolCall.status` (`acknowledged` vs `failed`) |
| `mcp.method.name`, `mcp.session.id` | tool-call metadata |
| event `gen_ai.evaluation.result` | write our scores back as events |

**Caveats to design around.** The GenAI conventions are pre-1.0; attribute names
change between versions. So: pin the semconv version in config, keep the mapping
in one file (`ingest/semconv.py`), and version the mapping itself. `committed`
status cannot come from a span — only the application knows whether state
actually changed, so services must emit a `tool.committed` attribute or we treat
the call as `acknowledged` only.

### Instrumentation contract for agent teams

Ship a small `evalkit-otel` helper so teams get the right spans without
learning semconv. One decorator for the agent loop, one for each tool. Document
it in `docs/instrumentation.md`. Adoption of the eval system is gated on this
being trivial.

---

## 29. Phase 2 — online evaluation in production

Offline catches regressions. Online catches drift and the failures nobody
anticipated.

### Pipeline

```
production trace
   → sampler (5–10%, stratified)
   → PII scrub
   → code graders        (free, milliseconds, every sampled trace)
   → judge               (sampled subset only, async, budgeted)
   → online_scores
   → threshold check → alert
   → failures + low-confidence → review queue
```

### Sampling policy

Not uniform random. Stratified so rare-but-important traffic is not invisible:

```yaml
sampling:
  base_rate: 0.05
  strata:
    - {match: "intent == 'refund'",        rate: 0.50}   # money moves
    - {match: "tool_error_count > 0",      rate: 1.00}   # every failure
    - {match: "escalated == true",         rate: 1.00}
    - {match: "turns > 8",                 rate: 0.30}   # long sessions fail more
    - {match: "language != 'en'",          rate: 0.25}   # under-tested slice
  max_traces_per_hour: 2000
  judge_subsample: 0.20                                   # of sampled traces
```

### Which graders run online

Only those that need no ground truth: `state.no_side_effects` (from the diff the
service reports), `flow.loop_detection`, `tools.arguments` schema validity,
`safety.*`, honesty checks, and the judge on answer quality. Graders requiring
`expected` are offline-only by definition.

### Alerting

- Online pass rate for any stratum drops >X points week-over-week
- Any `safety.*` critical failure → page immediately, no threshold
- Judge κ against the rolling gold set drops below floor → the *judge* is broken,
  not necessarily the agent; suppress quality alerts until recalibrated
- Volume of a stratum collapses → instrumentation broke, treat as an outage

### Offline↔online correlation

Track it explicitly. If the offline suite says 94% and online says 71%, the
suite no longer represents production and is not protecting anything. Reported
monthly; a persistent gap triggers a curation sprint.

---

## 30. Phase 2 — review queue and the data flywheel

This is the loop that makes the system improve rather than decay.

```
production failure
   → review queue (reason: failure | low_confidence | sampled | judge_disagreement)
   → human labels: correct outcome, forbidden actions, category
   → two destinations:
        gold_labels   → recalibrate the judge (κ)
        suites/*/cases → new regression case (after sanitisation + approval)
   → next run protects against it, forever
```

### Queue rules

- Every item carries the trace, the scores, and *why* it was queued
- Judge disagreements (Terra vs Luna) go straight to the queue — that mechanism
  already exists in `LLM_AS_JUDGE` and is reused verbatim
- Labelling is structured, not free text: expected final state, required calls,
  forbidden mutations, category. That is what a case needs
- Two labellers on a 10% overlap sample; inter-annotator agreement (κ) is
  tracked. If humans disagree, the rubric is ambiguous, not the labeller
- Promotion to a suite requires: sanitisation, `review_status=approved`, and an
  owner. `ef cases add-from-trace` does the mechanical part

### Dataset health, reported weekly

Age distribution of cases (`added_at`), share sourced from production vs
handwritten, coverage per intent and per language, saturation per suite,
and the count of cases retired for behaviours that no longer exist. A golden set
that stops resembling production stops predicting production.

---

## 31. Phase 2 — dashboard

Jinja + HTMX, served by the API. Two views only. Every panel must change a
decision; anything else is deleted.

**View 1 — Release comparison.** Candidate vs baseline: verdict banner, pass^k
with CI per suite, slice table sorted by regression size, the failure list with
drill-down to trajectory, cost and p95 latency deltas, and the manifest diff
(what actually changed between the two runs).

**View 2 — Production health.** Online pass rate by stratum over time with
sample counts, open review-queue depth, judge κ trend, offline↔online gap, and
alert history.

Distinguish *no data* from *good performance* — an empty panel must say "no
samples", never render as 100%.

---

## 32. Phase 3 — multi-tenant platform (only if needed)

Do not start this unless a second team is actively blocked without it. It is
6–9 months and most of it is not eval work.

What it requires beyond Phase 2:

1. **Stable public contracts** — `Adapter`, `Grader`, `Environment`, `Store`
   become semver-governed API with a deprecation policy and a **conformance test
   kit** third parties run to prove their implementation behaves.
2. **Plugin distribution** — entry-point discovery, or a registry of vetted
   grader packages. Custom graders from other teams must run sandboxed.
3. **Tenancy** — org/project scoping on every table, RBAC, per-project secrets,
   data segregation, audit log of who ran and who overrode what.
4. **Quotas and chargeback** — per-project token/cost budgets, rate-limit
   fairness, spend reporting.
5. **Hard isolation** — gVisor/Firecracker or per-tenant K8s namespaces with
   egress policy. A container alone is not a security boundary.
6. **Self-service onboarding** — suite templates, `ef init`, docs, and a worked
   example a new team can run in under an hour.

**Honest framing:** at this point Inspect AI is already the chassis and Langfuse
is already the trace layer. The defensible value we would add is the grader
library and the domain packs — not the runner. Revisit build-vs-adopt here
deliberately rather than by momentum.

---

## 33. Security and threat model

### Assets

Agent credentials, production traces (may contain customer PII), gold labels,
judge API keys, and the gate itself — because a compromised gate ships anything.

### Threats and controls

| Threat | Control |
|---|---|
| Eval touches real customer systems | `env/` is the only backend offline; no production credentials in any eval environment; `ef doctor` asserts the target is not a production host |
| Prompt injection in fixtures or traces reaches the judge | judge input is data, never instructions; delimit and label; injection probes in the judge's own regression suite |
| Malicious tool output escapes into grading logic | graders never `eval`/`exec`; all tool results treated as untrusted strings |
| Untrusted agent code (CLI adapter) | Docker with no network, read-only mounts, dropped capabilities, CPU/memory limits, no host credentials |
| Secret leakage into runs/reports | manifest stores hashes, not values; a secret-scan test asserts no `sk-`/`Bearer` patterns in any artifact |
| Gate bypass | gate runs in CI, not locally, on a protected branch; overrides require a labelled PR, a named approver, and are written into the run record |
| Trace store exfiltration | PII scrubbed at ingest; 30-day retention; access via service token; no public endpoint |
| Supply chain | `uv.lock` committed, pinned digests for CI images, Dependabot, no unpinned `latest` |

### Explicit non-goal for Phase 1

We are not defending against a malicious agent author. The CLI adapter is for
our own agents. If that changes, §32 item 5 becomes mandatory first.

---

## 34. Data governance, privacy, retention

- **Fixtures are synthetic.** Always. No production customer data in `suites/`.
- **Production traces are scrubbed at ingest**, before they hit Postgres: emails,
  phone numbers, order/customer identifiers, payment fragments, free-text names.
  Scrubbing is a versioned component with its own test fixtures, and its version
  is recorded on every trace row.
- **Promotion to a case requires sanitisation plus approval.** A trace becomes a
  case only after a human confirms no residual identifiers, replacing real ids
  with fixture ids.
- **Retention.** Production traces 30 days. Trajectories from eval runs 180 days
  hot, then cold. Runs, scores, and manifests indefinitely (they are small and
  they are the audit trail). Gold labels indefinitely.
- **Access.** Trace and review-queue access is logged. Eval results are readable
  by all engineers — the whole point is that they inform decisions.
- **Right-to-erasure.** Because traces are scrubbed and short-lived, and cases
  are synthetic after sanitisation, an erasure request is satisfied by deleting
  matching trace rows. Document the query; test it.
- **Audit.** Every gate decision is recorded: run id, verdict, reasons, approver
  if overridden. This is what makes the process defensible after an incident.

---

## 35. Cost model and budget control

### Where money goes

1. The **agent under test** — trials × cases × tokens. Dominant cost.
2. The **judge** — only on semantic dimensions, only on some suites.
3. The **simulated user** — multi-turn cases only.
4. Infrastructure — negligible in Phase 1, small in Phase 2.

### Worked estimate (illustrative)

| Run | Cases | Trials | Agent cost | Judge cost | Total |
|---|---|---|---|---|---|
| PR smoke | 50 | 1 | ~$0.50 | 0 (judge off) | **~$0.50** |
| Nightly full | 250 | 5 | ~$12 | ~$3 | **~$15** |
| Weekly + adversarial + heldout | 400 | 5 | ~$22 | ~$6 | **~$28** |
| Online (5% of 100k/mo) | 5,000 | — | 0 | ~$25 | **~$25/mo** |

≈ **$500–600/month** at this scale. Recompute with real token counts after
Week 5; the numbers above are shape, not measurement.

### Controls

- Per-case budget caps (tokens, cost, wall-clock); exceeding → `budget_exceeded`,
  which is a *recorded failure*, not a crash
- Per-run ceiling; the runner aborts and marks the run invalid rather than
  silently overspending
- Staged suites: cheap structural checks on PRs, expensive judgment nightly
- Judge cost tracked **separately** from agent cost in every report. When judge
  cost exceeds ~25% of the agent cost, cut the judge subsample or distil to a
  smaller judge
- Caching on for local iteration, **always off** for gated runs
- `cost per successful task` is a first-class reported metric, not just total
  spend — it is the number that actually compares two agent versions

---

## 36. Operating the eval system

The eval system is production software. It gets the same treatment.

### SLOs

| Signal | Target |
|---|---|
| PR smoke suite wall-clock | p95 < 5 min |
| Nightly suite completion | > 99% of scheduled runs complete |
| Invalid-run rate (crashes, missing data) | < 1% |
| Flaky-case rate (0 < successes < k) | < 5% of cases |
| Online scoring lag | p95 < 10 min from trace to score |
| Judge κ vs gold set | ≥ 0.70, alert below |

### Its own telemetry

The harness emits structured logs (`run_id` bound) and metrics: trials started /
completed / failed / invalid, grader latency, provider 429 rate, cost per run,
queue depth. **Evaluator health is monitored like any other service** — a
silently degrading judge is worse than no judge.

### Alerts

| Alert | Severity | Action |
|---|---|---|
| Nightly run failed to complete | high | runbook R1 |
| Invalid-run rate > 1% over 24h | high | runbook R2 |
| Judge κ below floor | high | runbook R3, suppress quality alerts |
| Online safety-critical failure | page | incident |
| Flaky-case rate > 5% | medium | runbook R4 |
| Offline↔online gap > 15 points | medium | curation sprint |
| Eval spend > 150% of budget | medium | review sampling and trials |

---

## 37. Runbooks

**R1 — Nightly run did not complete.**
Check provider status and 429 rate → check the agent staging deployment →
`ef run --resume <run_id>` → if it fails again on the same case, that case has an
environment or fixture bug; quarantine it (tag `quarantine`, excluded from the
gate, tracked) and file an issue. Never delete it.

**R2 — Invalid-run rate spiking.**
Group invalid trials by `stop_reason`. `timeout` → check concurrency vs provider
rate limits; we may be manufacturing failures. `error` → read the exception; it
is usually an adapter contract change. `budget_exceeded` → the agent regressed
into a loop; that is a real finding, not an infrastructure problem.

**R3 — Judge κ dropped.**
Do not trust any judge-derived metric until resolved. Re-run the judge stability
check (repeat + position swap). If stability is fine but κ dropped, the *gold
set* drifted or the rubric no longer matches the product — recalibrate on the
calibration split and verify on held-out. If stability degraded, the judge model
version changed; pin it.

**R4 — Flaky cases.**
Flakiness is either agent nondeterminism (real, and pass^k will show it) or
harness leakage (isolation, shared fixture, ordering). Run the case 20× serially
and 20× concurrently; if only concurrent runs fail, it is leakage — fix the env,
not the case.

**R5 — Gate blocked a release and the team disagrees.**
Read the failing trajectories first. Three legitimate outcomes: the agent is
genuinely wrong (fix it); the grader is too rigid (fix the grader, re-score the
baseline, note the version change); the case is ambiguous (rewrite the case,
because two experts must reach the same verdict). "Override and ship" is a
fourth outcome that requires a named approver and an issue.

**R6 — Production incident traced to the agent.**
Capture the trace → reproduce as a case in the local suite → confirm the current
suite does *not* catch it → add it as a regression case → fix → confirm the case
now passes → keep it forever. Every incident ends with a test.

---

## 38. Release management of the harness itself

- The harness is versioned (`evalkit` semver) and tagged. Runs record the version.
- **Breaking changes to `schema/`** are major bumps and require a documented
  migration for stored runs, because comparability depends on it.
- **Grader changes are the sensitive ones.** Any change to grading logic bumps
  the grader version hash, which invalidates comparison with prior runs. Process:
  change → re-score the current baseline as a new run → confirm the verdict is
  unchanged on known-good data → then adopt. This is written into the PR template.
- CHANGELOG entries for every grader and schema change, with the reason.
- The harness has its own CI: lint, mypy on `schema/`, full test suite, and a
  self-test that runs the demo suite against the echo adapter end-to-end.

---

## 39. Team, ownership, and rituals

Eval suites are living artifacts. Unowned suites rot within a quarter.

| Role | Responsibility |
|---|---|
| **Harness owner** (1 eng) | `src/evalkit`, CI, gate, releases |
| **Suite owner** (1 per suite) | cases, fixtures, rubric, thresholds; named in the suite README |
| **Labellers** (rotating) | review queue, gold set; 2 people minimum for κ |
| **Release approver** | signs gate overrides |

### Rituals

- **Weekly transcript review, 60 min.** The whole team reads ~20 trajectories
  and their grades, including passes. This is the only reliable way grader bugs
  get found. Non-negotiable; it is the single highest-leverage hour in the process.
- **Weekly dataset health** (5 min, automated report from §30).
- **Monthly saturation review.** Suites above 80% lose signal; graduate cases and
  add harder ones.
- **Monthly offline↔online correlation review.**
- **Post-incident:** every incident produces a case (R6).

---

## 40. Rollout: shadow → advisory → enforced

Do not turn on a blocking gate first. Turn it on last.

| Stage | Duration | Gate behaviour | Exit criterion |
|---|---|---|---|
| **1. Shadow** | 2 weeks | runs, reports, blocks nothing | < 5% flaky, < 1% invalid, graders reviewed by hand |
| **2. Advisory** | 2 weeks | posts PR comment, does not block | zero false blocks in a dry run against the last 20 merged PRs |
| **3. Enforced (critical only)** | 2 weeks | blocks on critical graders only | no override needed for a legitimate change |
| **4. Fully enforced** | ongoing | full gate, all stages | — |

**Stage 2's exit criterion is the important one.** Replay the gate against the
last 20 merged PRs. If it would have blocked a change that was actually fine,
the graders are too rigid and enforcement would destroy trust on day one. Trust
in the gate is the asset; it is spent instantly and rebuilt slowly.

---

## 41. Full timeline to production

| Phase | Weeks | Outcome |
|---|---|---|
| **Phase 1 — Harness** | 1–10 | §17. Local + CI, deterministic + judge graders, gate, report |
| **Rollout** | 11–16 | §40. Shadow → advisory → enforced |
| **Phase 2a — Persistence & API** | 13–17 | Postgres store, API, runs queryable over time |
| **Phase 2b — Trace ingestion** | 17–20 | OTLP ingest, instrumentation helper, agent teams emitting spans |
| **Phase 2c — Online evals** | 20–24 | Sampler, online graders, alerting, offline↔online correlation |
| **Phase 2d — Review queue & flywheel** | 22–26 | Labelling, gold set, `add-from-trace` in routine use |
| **Phase 2e — Dashboard** | 25–27 | Two views, replacing ad-hoc report sharing |
| **Steady state** | 28+ | Rituals from §39; suite expansion; second agent onboarded |
| **Phase 3 — Platform** | — | Only on evidence of a second blocked team (§32) |

Phases 2a–2e overlap with rollout deliberately: the gate must be trusted before
the platform around it is worth building.

**Milestones.**
M1 (wk 4) first real agent graded end-to-end ·
M2 (wk 6) gate blocks a deliberately bad change ·
M3 (wk 10) Phase 1 done, all §24 boxes ticked ·
M4 (wk 16) gate fully enforced on main ·
M5 (wk 24) production failures scored online and flowing into the suite ·
M6 (wk 27) dashboard is where release decisions get made.

---

## 42. Program success metrics

Measure the eval program, not just the agent. Reviewed quarterly.

| Metric | Target |
|---|---|
| Agent regressions caught **before** release | ≥ 90% of those found within 30 days |
| Production incidents with a regression case added | 100% |
| Median time from production failure → protected case | < 1 day |
| Share of assertions that are deterministic | ≥ 80% |
| Gate override rate | < 5% of releases |
| Offline↔online pass-rate gap | < 10 points |
| Engineer trust ("would you ship on a green gate?") | yes, informally surveyed |
| Eval spend as share of agent inference spend | < 10% |

The last qualitative one matters most. A gate people route around is worse than
no gate, because it manufactures false confidence.

---

## 43. Documentation deliverables

| Doc | Audience | When |
|---|---|---|
| This README | everyone | done |
| `docs/adding-a-suite.md` | agent teams | Week 10 |
| `docs/writing-cases.md` — unambiguity, balance, reference solutions | suite owners | Week 3 |
| `docs/graders.md` — every grader, what it proves, what it does not | everyone | Week 3, updated per grader |
| `docs/adapter-contract.md` — the HTTP contract | agent teams | Week 4 |
| `docs/interpreting-results.md` — CIs, pass^k, what a delta means | everyone | Week 6 |
| `docs/instrumentation.md` — OTel spans for online evals | agent teams | Week 18 |
| `docs/runbooks.md` — §37 expanded | on-call | Week 12 |
| `suites/*/README.md` — what it proves **and does not prove** | suite owners | per suite |

The last one is a hard requirement. A suite that does not state its limitations
will be over-trusted.

---

## 44. Open decisions

Resolve before the week they block. Each has a default so nothing stalls.

| # | Decision | Default if undecided |
|---|---|---|
| 1 | Which agent is the first real target (Week 4)? | the support agent used throughout this plan |
| 2 | Judge provider pair | Claude Opus 5 + GPT-5, different families, per §7.9 |
| 3 | Trajectory blob storage in Phase 2 | filesystem first, S3/MinIO when >100 GB |
| 4 | Keep Azure OpenAI as the judge transport | yes — it already works; add providers alongside |
| 5 | Object store vs Postgres for traces | traces in Postgres (scrubbed, 30 days), trajectories on disk |
| 6 | `evalkit` as its own repo or a package in this one | this repo, `src/` layout, split only if a second consumer appears |
| 7 | Trials for the PR smoke suite | 1 (cost); nightly carries the reliability signal |
| 8 | Who owns the gold set | the suite owner, with 2 rotating labellers |

## References

- Anthropic — *Demystifying evals for AI agents* (grader types, pass^k, task
  design, eval saturation, grader hacking)
- τ-bench / τ²-bench (arXiv 2406.12045) — database-state grading, simulated
  users, pass^k
- Inspect AI, UK AISI — Task/Dataset/Solver/Scorer decomposition; harness
  outside the sandbox
- Chen et al. 2021 (arXiv 2107.03374) — unbiased pass@k estimator
- *Don't use the CLT in LLM evals with fewer than a few hundred datapoints*
  (arXiv 2503.01747)
- OpenTelemetry GenAI semantic conventions — trajectory schema alignment
  (pre-1.0; pin the version)
- OWASP Top 10 for Agentic Applications — adversarial suite taxonomy
- MCP-Bench (Accenture, ICLR 2026), MCPAgentBench — MCP evaluation with
  distractor tools
- *AI Evals Engineering Handbook*, 228 topics — the conceptual foundation this
  plan implements
- `/Users/ankitraj/Developer/evals/LLM_AS_JUDGE` — the judge subsystem this
  plan reuses
