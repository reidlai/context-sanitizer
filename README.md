# context-sanitizer

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-reidlai-ea4aaa?logo=github-sponsors)](https://github.com/sponsors/reidlai)

**context-sanitizer** is a blazingly fast, local Rust proxy and library that intercepts LLM API traffic to act as a **System One semantic firewall** and token compressor.

Built for modern Agentic AI and Copilot workflows, it solves the "context rot" problem by trimming stale history and massive tool schemas, while ensuring your AWS keys, GitHub tokens, and proprietary semantic data never leak to third-party LLM providers. It offers a pure Rust core with seamless Python bindings via `PyO3`.

## The 3-Tier Hybrid Architecture

Built specifically to sit at the boundary of **Copilot UIs (including CopilotKit and OpenAI-compatible APIs)** and **Agentic AI frameworks**, `context-sanitizer` executes a **3-Tier Speculative Cascade** for sub-millisecond speed out-of-the-box and deep neural accuracy when enabled:


```text
Incoming CopilotUI / OpenAI API / Agent Payload
   │
   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Tier 1 (L0): Pure-Rust Deterministic & Lexical Guard (< 500,000 ns)       │
│  • aho-corasick SIMD secret & PII detection (Root Keys -> [REDACTED_CREDENTIAL])│
│  • Session-Scoped Shadow Vault (<SHADOW:EMAIL_8f3a2c> + .rehydrate() API) │
│  • Pure-Rust lexical/JSON scanner (logos + serde_json) syntax token lock  │
└───────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Tier 2 (L0.5): Pure-Rust Statistical Pruning (< 1,500,000 ns | 0 MB)      │
│  • Zero-weight information-theoretic pruning (tiktoken-rs + BM25/TF-IDF)  │
│  • Tail-query auto-anchoring, JSON schema minification & history dedup    │
└───────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Tier 3 (L1): Configurable Trait-Based System One ML (--features ml)       │
│  • Pluggable traits (SemanticGuard + TokenCompressor) & HF model_id       │
│  • Stage 3A: Zero-Shot NLI / Prompt Guard circuit breaker (short-circuits)│
│  • Stage 3B: LLMLingua-2 BERT token classifier (runs on allowed payloads) │
└───────────────────────────────────────────────────────────────────────────┘
```

1. **The Configurable System One Gatekeeper (`SemanticGuard` Trait):** Uses blazing-fast, open-source System One encoder models (defaulting to *DeBERTa-v3 Zero-Shot NLI*, swappable via `guard_model_id` to *ModernBERT*, *Llama-Prompt-Guard*, or custom ONNX/Candle models) to block malicious code injections and proprietary semantic leaks in a single forward pass—short-circuiting before compression runs on blocked threats.
2. **The Session-Scoped "Shadow Vault" + Split Secret Policy:** Operational PII (emails, IPs, phones) is swapped with session-scoped cryptographic shadow tokens (e.g., `<SHADOW:EMAIL_8f3a2c>`) bound to a `VaultHandle` with a destination-whitelisted `.rehydrate(args, destination)` API so agent tool calls keep working, while root credentials (AWS keys, GitHub PATs) default to one-way `[REDACTED_CREDENTIAL]` masks so indirect prompt injections can never exfiltrate keys.
3. **The 100% Pure-Rust & ML Compression Engine (`TokenCompressor` Trait):** Leverages zero-C pure-Rust lexical syntax locking (`logos` + `serde_json`) and statistical pruning by default (0 MB model download required), plus optional neural token classification (*LLMLingua-2*, swappable via `compression_model_id`) to shrink payloads token-by-token without ever breaking code or JSON syntax.

## Research Foundation

This project implements state-of-the-art token pruning and semantic architectures based on recent publicly available research in context engineering:

* **System One Encoder Gatekeepers (*DeBERTaV3*, He et al., ICLR 2023; *Meta Llama Prompt Guard 2*, 2025; *ModernBERT*, Warner et al., 2024)**: Shifts boundary safety away from slow sequential System 2 token generation (`arXiv:2111.09543`, `arXiv:2412.13663`). Non-autoregressive, bidirectional encoder architectures (`microsoft/deberta-v3-base`, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, `meta-llama/Llama-Prompt-Guard-2-22M` / `86M`, and `answerdotai/ModernBERT-base`) compute text semantics and detect indirect prompt/code injections in a single forward pass.
* **LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression (Pan et al., ACL Findings 2024, `arXiv:2403.12968`)**: Microsoft Research's framework formulates prompt compression as a binary token classification problem ($p(\text{preserve})$). Using a bidirectional Transformer encoder (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`), it achieves 3x–6x faster compression than perplexity-based methods and accelerates end-to-end latency by 1.6x–2.9x while preserving downstream accuracy.
* **Zero-Weight Information-Theoretic & Submodular Pruning (*Selective Context*, Li et al., EMNLP 2023; *RECOMP*, Xu et al., ICLR 2024; *Lin & Bilmes*, ACL 2011; *Robertson & Zaragoza*, 2009)**: Inspired by *Selective Context* (`arXiv:2310.06201`, which uses causal LM conditional self-information $I(x_t) = -\log_2 P(x_t \mid x_{<t})$) and *RECOMP* (`arXiv:2310.04408`, which uses trained compressors for selective RAG augmentation), our Pure-Rust Tier 2 engine replaces neural weights with **static unigram Shannon self-information / IDF** ($I(w) = -\log P_{\text{unigram}}(w)$), **Okapi BM25**, and **budgeted monotone submodular chunk selection** (Lin & Bilmes, ACL 2011, with a $(1 - 1/e)$ greedy guarantee) to prune low-entropy filler tokens with 0 MB of GPU/CPU model weights.
* **Prompt Compression for Large Language Models: A Survey (Zongqian Li et al., 2024 / NAACL 2025, `arXiv:2410.12388`)**: Establishes the academic taxonomy of prompt compression, dividing the field into "hard" prompt compression (discrete token filtering/pruning) and "soft" prompt compression (trainable continuous embedding vectors).

## Development Methodology

This project is engineered using **Spec-Driven Development (SDD)** guided by the **BMAD Method**.

Rather than ad-hoc implementation, all architecture and capabilities follow a disciplined, specification-first engineering lifecycle:

* **Specification-First Architecture**: Interface contracts, C-ABI boundaries (`PyO3`), PII/secret regex invariants, and System One guardrails are strictly formalized before code implementation.
* **Bounded Multi-Agent Execution**: Complex pipelines (such as hybrid lexical pruning vs. neural classification) are decomposed into discrete, bounded tasks with explicit validation criteria.
* **Deterministic Verification & 6-Stage DevSecOps Pipeline**: Enforces a two-speed local `.pre-commit-config.yaml` hook suite (`pre-commit` fast lint/SAST/SCA checks + `pre-push` unit/integration/coverage gates) and a 6-stage GitHub Actions pipeline: (1) Formatting & Linting (`cargo fmt`, `clippy`, `ruff`), (2) Source Code SAST & Secret Scanning (`#![forbid(unsafe_code)]`, panic-free lints, Semgrep/CodeQL, `gitleaks`), (3) Software Composition Analysis (`cargo-audit`, `cargo-deny`, `pip-audit`), (4) Pre-Build Unit Tests $\to$ Source Integration Tests $\to$ $\ge 85\%$ Code Coverage Gate (`cargo-llvm-cov` & `pytest-cov`), (5) Gated Artifact Compilation (`cargo build --release`, `maturin`, `wasm-pack`), and (6) Post-Build Black-Box Artifact E2E Integration & Benchmark Verification.

## Features

* **System One Semantic Firewall & Code-Injection Gatekeeper**: Intercepts malicious code injections, jailbreaks, and proprietary semantic leaks entering through Copilot UIs (CopilotKit, OpenAI API `/v1/chat/completions`) before they reach your agentic framework.
* **Pluggable Model Traits (`SemanticGuard` & `TokenCompressor`)**: Ships with *DeBERTa-v3 Zero-Shot NLI* and *LLMLingua-2 BERT* defaults, while allowing users to swap in any Hugging Face or local checkpoint (`guard_model_id` / `compression_model_id`, e.g., *ModernBERT*, *Llama-Prompt-Guard*, or custom ONNX/Candle models) or implement the Rust traits directly.
* **Session-Scoped "Shadow Vault" + Split Secret Policy**: Replaces operational PII (emails, phone numbers, IPs, customer IDs) with session-scoped cryptographic tokens (`<SHADOW:EMAIL_8f3a2c>`) that can be rehydrated locally via `vault.rehydrate(args, destination="...")` for whitelisted tool calls, while root credentials (AWS keys, GitHub PATs, private keys) default to one-way `[REDACTED_CREDENTIAL]` masks so indirect prompt injections can never exfiltrate keys.
* **3-Tier Hard Token Compression (100% Pure-Rust & Syntax-Safe)**: Combines a zero-C pure-Rust lexical/JSON scanner (`logos` + `serde_json`) for syntax token locking with Pure-Rust statistical pruning (Tier 2: `tiktoken-rs` + BM25/self-information with tail-query auto-anchoring) and *LLMLingua-2* neural token classification (Tier 3).
* **Context X-Ray CLI & Benchmark Visualizer (`context-sanitizer xray`)**: Built-in terminal diagnostic tool (`context-sanitizer-cli`) that visualizes pruned tokens, Shadow-Vaulted PII, one-way redacted secrets, System One threat scores, dollar savings, and nanosecond (`ns`) latency (`std::time::Instant::elapsed().as_nanos()`).
* **Zero-Leak SIMD Security & Anti-ReDoS Linear-Time Guarantee**: Synchronous, fail-closed interception of hardcoded secrets and PII before network dispatch using the `aho-corasick` SIMD DFA and `logos` DFA lexers—enforcing strict $O(N)$ linear-time scanning with zero backtracking (`~5–25 ns/byte`, `~20,000–100,000 ns` median per 4KB, `< 500,000 ns` P99 ceiling even on adversarial ReDoS payloads).
* **Modular 3-Crate Cargo Workspace**: Clean separation across `context-sanitizer` (core Rust library), `context-sanitizer-cli` (`xray` binary), and `context-sanitizer-py` (`PyO3` Python bindings).

## Context X-Ray CLI (`context-sanitizer xray`)

Inspect, benchmark, and visualize exactly how `context-sanitizer` protects and compresses your prompts in the terminal:

```bash
cargo install context-sanitizer-cli
context-sanitizer xray payload.json --rate 0.35
```


```text
╭─ Context X-Ray: payload.json ────────────────────────────────────────────────╮
│ Tier 1 (L0 Guard)    : 2 PII Vaulted | 1 Root Key Redacted | 14 syntax locks │
│ Tier 2 (L0.5 Rust)   : Pruned 1,420 low-entropy & schema bloat tokens        │
│ Tier 3 (L1 System 1) : Threat Score 0.03 (SAFE) | Route: Code+Chat           │
├──────────────────────────────────────────────────────────────────────────────┤
│ User: Please check login failure for <SHADOW:EMAIL_8f3a2c> on <SHADOW:IP_9b1e>│
│ Code: AWS_ACCESS_KEY_ID=[REDACTED_CREDENTIAL]                                │
│ function execute_deployment() {                                              │
│   [-// This is a very long comment that explains nothing-]                   │
│   return true;                                                               │
│ }                                                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ Tokens : 3,200 ──► 1,088 (66.0% saved | Target: 0.35)                        │
│ Savings: ~$0.0318 / 1k calls (GPT-4o)  |  Latency: 412,000 ns (L0+L0.5)      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Reproducible Benchmarks & Adversarial Latency SLAs (`--bench`)

Run deterministic throughput, compression, and adversarial anti-DoS benchmarks locally (with 0 MB of downloaded model weights required for Tier 1 + Tier 2):

```bash
context-sanitizer xray payload.json --bench
# Or run the Criterion microbenchmark suite:
cargo bench
```

| Tier / Benchmark Suite | Algorithmic Complexity | Median Latency (4KB & Per-Byte) | P99 Ceiling (Adversarial Inputs) | Security & Quality Invariant |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (`L0` Deterministic Guard & Vault)** | $O(N)$ linear-time (`aho-corasick` SIMD DFA, zero backtracking) | `~20,000–100,000 ns` (`~5–25 ns/byte`) | **`< 500,000 ns`** (`< 120 ns/byte`) on adversarial ReDoS payloads | 100% credential interception; `Redacted` & `Shadow` spans override `Locked` spans (blocks syntax-lock smuggling) |
| **Tier 2 (`L0.5` Statistical Compressor)** | $O(N \log N)$ lazy-greedy max-heap submodular selection + `logos` DFA | `~300,000–400,000 ns` (`~75–100 ns/byte`, `~412,000 ns` combined L0+L0.5) | **`< 1,500,000 ns`** (`< 365 ns/byte`) on adversarial "compression bomb" payloads | 40%–70% token reduction (`MeetingBank`/`GSM8K` within `1.5%` accuracy); $\ge 30\%$ tool schema minification; `0%` JSON parse errors (Fail-Open AST invariant) |
| **Tier 3 (`L1` System One Gatekeeper, `--features ml`)** | `< 100,000 ns` $O(N)$ SIMD fast-breaker + Top-$K$ ($K \le 2$) surprisal-guided 512-token window (`candle-transformers`) | `< 100,000 ns` (Fast-Breaker, `~10–25 ns/byte`) / `~1,500,000–3,500,000 ns` (22M Quantized CPU) | **`< 5,000,000 ns`** hard `guard_timeout_ns` deadline (blocks Neural Sponge DoS) | Blocks delimiter spoofing & markdown exfiltration in `< 100,000 ns`; short-circuits semantic injections (`>= 0.85`); zero-crash fallback to Tier 2 |


## Quick Start: Python

The Python package (`context-sanitizer-py`) is pre-compiled using `maturin` with zero C/C++ grammar dependencies. You do not need Rust installed on your machine to use it.

### Installation

```bash
pip install context-sanitizer-py
```

### Usage

```python
import context_sanitizer as cs
from context_sanitizer.integrations import LangChainSanitizer, CrewAISanitizer, AutoGenSanitizer

# Initialize the engine with security, Shadow Vault, and configurable models
sanitizer = PromptSanitizer(
    use_system_one_guard=True,         # Enable local System One semantic firewall & routing
    use_ml_compression=True,           # Enable neural compression (Stage 3B)
    guard_model_id="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", # Configurable!
    compression_model_id="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    target_compression_rate=0.33,      # Target rate: keep ~33% of the original tokens
    force_tokens=['\n', '?']           # Ensure structural and interrogative tokens are never dropped
)

dirty_prompt = """
System: You are an AI assistant.
User: Please check why alice@acme.com is getting an error during deployment.
Code: 
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
function execute_deployment() {
    //  This is a   very    long     comment that explains nothing
    return true;
}
"""

# Scrubs root keys ([REDACTED_CREDENTIAL]), vaults PII (<SHADOW:EMAIL_*>), compresses bloat, and runs System One guardrails
sanitized_messages, vault = cs.sanitize(dirty_prompt, rate=0.33, allowed_destinations=["lookup_user"])

# Handle security block or proceed with clean, compressed prompt
if result.get("status") == "blocked":
    print(f"⚠️ Security Alert: Payload blocked! Reason: {result['reason']}")
    print(f"Violations detected: {result.get('violations', [])}")
else:
    print("--- Compressed Prompt (Safe for Cloud LLM) ---")
    print(result["compressed_prompt"])

    # When the LLM emits a tool call using the shadow token, rehydrate locally using the session handle:
    llm_tool_call = '{"tool": "lookup_user", "email": "<SHADOW:EMAIL_8f3a2c>"}'
    local_tool_args = vault.rehydrate(llm_tool_call, destination="lookup_user")
    print("\n--- Rehydrated Local Tool Call ---")
    print(local_tool_args)

    print("\n--- Optimization Metrics ---")
    print(f"Original Tokens: {result['origin_tokens']}")
    print(f"Compressed Tokens: {result['compressed_tokens']}")
    print(f"Compression Ratio: {result['ratio']} (Targeted 0.33)")
    print(f"Estimated API Savings: {result['saving']}")
```

## Quick Start: Rust

You can embed the pure Rust core directly into your own proxy servers or Rust-based AI agents, and swap models via builder configuration or the `SemanticGuard` and `TokenCompressor` traits.

### Installation

```bash
cargo add context-sanitizer
```

### Usage

```rust
use context_sanitizer::{SanitizerEngine, Decision};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configurable builder (swap Hugging Face or local model_id anytime)
    let engine = SanitizerEngine::builder()
        .with_target_compression_rate(0.35)
        .with_guard_model("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        .with_compressor_model("microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
        .build()?;
    
    let dirty_payload = r#"
        {
            "role": "user",
            "content": "My GitHub token is <YOUR_GITHUB_TOKEN>. Check account alice@acme.com please."
        }
    "#;
    
    // Inspects payload: root keys -> [REDACTED_CREDENTIAL], PII -> session Shadow Vault, syntax-locked compression
    match engine.inspect(dirty_payload) {
        Decision::Allowed(safe_payload) => {
            println!("Safe Payload for LLM:\n{}", safe_payload.content);

            // Rehydrate session-scoped PII tokens locally when executing tool calls
            let tool_args = r#"{"email": "<SHADOW:EMAIL_8f3a2c>"}"#;
            let rehydrated = safe_payload.vault_handle().rehydrate(tool_args, "db_tool")?;
            println!("Rehydrated Tool Args:\n{}", rehydrated);
        }
        Decision::Blocked(violation) => {
            eprintln!("⚠️ Security Alert: Payload blocked! [{}] {}", violation.category, violation.reason);
        }
    }

    Ok(())
}
```

## Agentic AI & Copilot UI Integrations

Modern Agentic workflows and **Copilot UIs (including CopilotKit and OpenAI-compatible APIs)** suffer from **context rot** (token bloat from recursive loops and verbose tool schemas), **user PII/secret exposure** (users pasting personal information or credentials into chat), and **adversarial code injection** (attackers injecting malicious payloads through frontend copilot widgets). 

`context-sanitizer` provides a **hardened two-tier defense model with a split secret policy**:
1. **Soft Mitigation (Session-Scoped Shadow Vault & Root-Secret Masking):**
   - **Operational PII** (emails, phones, IPs, user IDs) is swapped in-place with session-scoped cryptographic tokens (`<SHADOW:EMAIL_8f3a2c>`). The cloud LLM never sees raw PII, while local tool calls rehydrate the original values via `.rehydrate()` using the session's `VaultHandle`.
   - **Root Credentials** (AWS secret keys, GitHub PATs, private keys) default to one-way `[REDACTED_CREDENTIAL]` masks so an indirect prompt injection can never trick an agent tool into exfiltrating root keys.
2. **Hard Mitigation (Circuit Breaker & Quarantine):** When the System One gatekeeper (`SemanticGuard`) or pure-Rust lexical scanner detects malicious code injections, proprietary semantic leaks, or jailbreaks, it short-circuits execution immediately—protecting your infrastructure and preventing costly third-party API calls.

| Scenario | System Action | Downstream LLM Invoked? |
| :--- | :--- | :--- |
| **Operational PII in Copilot UI** (`email`, `IP`, `phone`) | Swapped via **Session-Scoped Shadow Vault** (`<SHADOW:EMAIL_8f3a2c>`) | **Yes** (Safe payload forwarded; tool calls rehydrated via `.rehydrate()`) |
| **Root Credentials** (`AWS_KEY`, `GITHUB_TOKEN`, `PEM`) | Masked one-way with `[REDACTED_CREDENTIAL]` tokens | **Yes** (Safe payload forwarded; root keys never rehydrated to tools) |
| **Token Bloat / Stale History / Tool Schemas** | Compressed via Pure-Rust Tier 2 & *LLMLingua-2* Tier 3 | **Yes** (Latency & cost reduced by 40–70%) |
| **Proprietary Semantic Leaks** | Blocked via Zero-Shot NLI + L0 custom terms | **No** (Short-circuited locally before API call) |
| **Malicious Code / Prompt Injections** | Quarantined to SecOps/SIEM | **No** (Zero inference cost incurred) |

---

### Python: LangGraph Integration (with Security Quarantine & Tool Rehydration)

Use LangGraph conditional edges (`add_conditional_edges`) to route execution to a `quarantine` node when severe security threats or semantic leaks are intercepted, and rehydrate Shadow-Vaulted PII locally before tool execution:

```python
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
import context_sanitizer as cs
from context_sanitizer.integrations import LangChainSanitizer, CrewAISanitizer, AutoGenSanitizer

# 1. Define Agent State with Security Flags & Vault Session ID
class AgentState(TypedDict):
    messages: List[BaseMessage]
    clean_prompt: Optional[str]
    vault_session_id: Optional[str]
    is_blocked: bool
    rejection_reason: Optional[str]

# 2. Initialize with Configurable System One Guardrails
sanitizer = PromptSanitizer(
    use_system_one_guard=True,
    guard_model_id="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    blocked_categories=["PROPRIETARY_IP", "UNRELEASED_FINANCIALS", "PROMPT_INJECTION"],
    hard_block_on_threat=True,
    target_compression_rate=0.40
)

# 3. Node: Sanitize, Vault PII, Compress, and Evaluate Threat Level
def sanitize_node(state: AgentState):
    raw_context = "\n".join(f"{m.type}: {m.content}" for m in state["messages"])
    result = sanitizer.sanitize_and_compress(raw_context)
    
    # Check if System One Firewall tripped a hard block (short-circuited)
    if result.get("status") == "blocked":
        return {
            "is_blocked": True,
            "rejection_reason": result["reason"]
        }
        
    return {
        "clean_prompt": result["compressed_prompt"],
        "vault_session_id": result["session_id"],
        "is_blocked": False
    }

# 4. Node: Security Quarantine (Circuit Breaker - No LLM Call)
def quarantine_node(state: AgentState):
    # Log to SIEM / SecOps and safely alert user without API invocation
    alert_msg = f"Security Policy Violation: {state['rejection_reason']}"
    return {"messages": state["messages"] + [AIMessage(content=alert_msg)]}

# 5. Node: Standard Model Execution + Local Tool Call Rehydration
def call_model_node(state: AgentState):
    response = llm.invoke(state["clean_prompt"])
    # If the LLM emitted tool calls containing <SHADOW:*> tokens, rehydrate locally:
    if getattr(response, "tool_calls", None) and state.get("vault_session_id"):
        for call in response.tool_calls:
            call["args"] = vault.rehydrate(call["args"], destination=call["name"])
    return {"messages": state["messages"] + [response]}

# 6. Build Conditional State Graph
def route_after_sanitization(state: AgentState) -> str:
    return "quarantine" if state["is_blocked"] else "agent"

workflow = StateGraph(AgentState)
workflow.add_node("sanitize", sanitize_node)
workflow.add_node("quarantine", quarantine_node)
workflow.add_node("agent", call_model_node)

workflow.set_entry_point("sanitize")
workflow.add_conditional_edges(
    "sanitize",
    route_after_sanitization,
    {
        "quarantine": "quarantine",
        "agent": "agent"
    }
)
workflow.add_edge("quarantine", END)
workflow.add_edge("agent", END)

app = workflow.compile()
```

---

### Rust: Rig Integration (Circuit Breaker & Shadow Vault Pattern)

In Rust-based agents built with [Rig](https://github.com/0xPlaygrounds/rig), use `Decision` matching to prevent tainted or malicious tool outputs from reaching the LLM and `.rehydrate()` to restore operational PII for local tools:

```rust
use context_sanitizer::{SanitizerEngine, Decision, SecurityViolation};
use rig::providers::openai::{self, GPT_4O};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai::Client::from_env();
    let engine = SanitizerEngine::builder()
        .with_guard_model("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        .build()?;

    // Raw payload from runtime tool execution containing confidential data
    let dirty_tool_output = r#"
        System: Executing cloud audit for customer alice@acme.com...
        Tool [bash]: AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
        Tool [files]: Confidential Q4 Unreleased Financial Forecasts for Project Alpha.
    "#;

    // Evaluate input through the System One firewall, Shadow Vault, & secret engine
    match engine.inspect(dirty_tool_output) {
        Decision::Allowed(clean_payload) => {
            // Forward sanitized & compressed context to Rig agent
            let agent = client
                .agent(GPT_4O)
                .preamble("You are a secure cloud infrastructure assistant.")
                .build();

            let response = agent.prompt(&clean_payload.content).await?;
            // Rehydrate any <SHADOW:*> operational PII tokens if needed for local follow-up tools:
            let local_output = clean_payload.vault_handle().rehydrate(&response, "cloud_audit_tool")?;
            println!("Rig Agent Response:\n{}", local_output);
        }
        Decision::Blocked(violation) => {
            // Hard circuit breaker triggered: DO NOT call the LLM
            eprintln!(
                "CRITICAL SECURITY ALERT: Blocked violation [{}] -> {}",
                violation.category, violation.reason
            );

            // Log incident to SIEM / abort execution safely
            return Err(format!("Halted: {}", violation.reason).into());
        }
    }

    Ok(())
}
```

## Build from Source

The repository is structured as a **3-crate Cargo workspace** (`100% pure Rust` by default with zero C/C++ grammar build dependencies):
* `crates/context-sanitizer`: Core Rust library (`L0` SIMD/lexical guard + Shadow Vault, `L0.5` statistical pruning, and `L1` `SemanticGuard` / `TokenCompressor` traits).
* `crates/context-sanitizer-cli`: The `context-sanitizer xray` terminal visualizer & benchmark binary.
* `crates/context-sanitizer-py`: `PyO3` / `maturin` Python bindings.

### Prerequisites

* **Rust**: Stable 1.75+ via [rustup](https://rustup.rs/)
* **Python**: 3.9+ (development headers and virtual environment)
* **cargo-maturin** *(Optional)*: `cargo install maturin` for packaging and editable installs

### 1. Clone the Repository

```bash
git clone https://github.com/reidlai/context-sanitizer.git
cd context-sanitizer
```

### 2. Build via Cargo

Compile the entire workspace (Rust library, proxy binary, and PyO3 extension) in release mode:

```bash
cargo build --release --workspace
```

Run the test suite across all crates:

```bash
cargo test --workspace
```

*(Optional)* Compile with local neural/ML compression features enabled:

```bash
cargo build --release --workspace --features ml
```

### 3. Build & Package Python Extension

You can package and install the Python bindings via the `cargo maturin` subcommand or compile directly using native `cargo build`:

#### Method A: Using `cargo maturin` (Recommended)

```bash
# Install cargo-maturin subcommand (one-time setup)
cargo install maturin

# Compile and link directly into your active Python environment
cargo maturin develop --release

# Or build standalone distributable wheels (.whl)
cargo maturin build --release
```

#### Method B: Pure `cargo build` (No Python build tooling)

Because PyO3 compiles to a standard C-ABI dynamic library (`cdylib`), you can build the extension using standard Cargo:

```bash
cargo build --release --features python
```

Copy or symlink the compiled shared object to your working directory or `PYTHONPATH`:
* **Linux:** `cp target/release/libcontext_sanitizer.so ./context_sanitizer.so`
* **macOS:** `cp target/release/libcontext_sanitizer.dylib ./context_sanitizer.so`
* **Windows:** `copy target\release\context_sanitizer.dll .\context_sanitizer.pyd`

## Contributing

Contributions are welcome! Direct pushes to the `main` branch are restricted to maintain codebase stability and strict security invariants. All contributors must follow the standard Fork & Pull Request workflow.

### Pull Request Workflow

1. **Fork the Repository:** Create your personal fork at [github.com/reidlai/context-sanitizer](https://github.com/reidlai/context-sanitizer).
2. **Create a Feature Branch:**
   ```bash
   git clone https://github.com/<your-username>/context-sanitizer.git
   cd context-sanitizer
   git checkout -b feature/your-feature-name
   ```
3. **Install Local `pre-commit` Hooks & Validate Changes:** Ensure all local DevSecOps hooks, formatting, lints, SAST/SCA checks, unit/integration tests, and $\ge 85\%$ coverage gates pass:
   ```bash
   cargo test --workspace
   cargo fmt --check
   pre-commit install --hook-type pre-commit --hook-type pre-push
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo deny check advisories bans licenses sources
   cargo audit
   cargo llvm-cov --workspace --all-targets --fail-under-lines 85
   ```
4. **Submit a Pull Request (PR):**
   * Push your feature branch to your personal fork.
   * Open a PR against `reidlai/context-sanitizer:main`.
   * Detail your changes, rationale, and verification steps in the PR description.
5. **Review & Merge:**
   * Every PR will be inspected by an AI review agent and the repository maintainer for correctness, security bounds, and test fidelity.
   * Feedback will be shared in the PR review comments, and the branch will be merged once approved.

### Reporting Issues

Encountered a bug, security vulnerability, or have an idea for a new feature? Please [open an issue](https://github.com/reidlai/context-sanitizer/issues/new/choose) using our standardized GitHub Issue Templates:

* 🐛 **[Bug Report](https://github.com/reidlai/context-sanitizer/issues/new?template=bug_report.yml):** For reporting unexpected panics, compression defects, or runtime crashes.
* 💡 **[Feature Request](https://github.com/reidlai/context-sanitizer/issues/new?template=feature_request.yml):** For proposing new regex detectors, System One encoder models, or agent framework integrations.
* 🛡️ **[Security Vulnerability](https://github.com/reidlai/context-sanitizer/issues/new?template=security_report.yml):** For reporting secret detector bypasses, false negatives in proprietary data detection, or guardrail escapes.

## Support & Sponsorship

If you find context-sanitizer valuable for securing your LLM pipelines or saving context tokens, consider supporting ongoing development:

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor%20on%20GitHub-reidlai-ea4aaa?style=for-the-badge&logo=github-sponsors&logoColor=white)](https://github.com/sponsors/reidlai)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### Third-Party Dependency Licenses (`NFR-7`)

All workspace dependencies use commercially permissive open-source licenses verified in CI via [`deny.toml`](deny.toml):

| Crate | Version | License | Role |
| :--- | :--- | :--- | :--- |
| `aho-corasick` | `1.1.5` | `Unlicense OR MIT` | Tier 1 $O(N)$ SIMD DFA pattern matching |
| `serde` | `1.0.210` | `MIT OR Apache-2.0` | Wire protocol serialization framework |
| `serde_json` | `1.0.128` | `MIT OR Apache-2.0` | OpenAI JSON parsing and validation |
| `thiserror` | `1.0.64` | `MIT OR Apache-2.0` | Unified library error derivations (`SanitizerError`) |
| `getrandom` | `0.2` | `MIT OR Apache-2.0` | 128-bit CSPRNG session salts for `VaultHandle` |
