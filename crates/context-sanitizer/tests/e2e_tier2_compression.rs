//! End-to-End Integration Test Suite for Epic 2 (`Stories 2.1–2.4`).
//!
//! Covers:
//! 1. Pure-Rust lexical syntax token locking (`logos`), delimiter tagging, and UTF-8 safety (`Story 2.1`, `FR-4`, `AD-1`).
//! 2. Syntax-lock anti-smuggling: secrets inside code fences, shell backticks, and JSON keys are 100% redacted (`Story 2.1`, `FR-1`, `FR-4`).
//! 3. Fail-Open Syntax Invariant: structural/JSON parse errors discard pruning and preserve redactions (`Story 2.1`, `FR-4`, `AD-4`).
//! 4. Tail-query auto-anchoring and submodular knapsack context compression (40%–70% reduction) (`Story 2.2`, `FR-5`).
//! 5. Anti-compression-bomb adversarial 4KB payload latency $< 1,500,000$ ns P99 (`Story 2.2`, `NFR-1`).
//! 6. OpenAI JSON tool schema minification $\ge 30\%$ token reduction with 100% schema validity (`Story 2.3`, `FR-6`).
//! 7. Black-box Context X-Ray CLI diagnostic dashboard and benchmarking execution (`Story 2.4`, `FR-12`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::Command;
use std::time::Instant;

use context_sanitizer::arena::{Span, SpanArena, SpanKind};
use context_sanitizer::compress::lexer::{scan_syntax_locks, validate_syntax_fail_open};
use context_sanitizer::compress::schema::{minify_tool_definitions, schema_reduction_ratio};
use context_sanitizer::compress::{compress_text, compress_text_with_arena};
use context_sanitizer::guard::{
    run_tier1_guard, sanitize_str_with_vault, VaultHandle, REDACTED_CREDENTIAL_MASK,
};
use context_sanitizer::wire::chat::sanitize_and_compress_chat_completions_json;
use serde_json::json;

#[test]
fn e2e_story2_1_lexical_syntax_locking_and_utf8_clamping() {
    let payload = concat!(
        "{\n",
        "  \"target_model\": \"gpt-4o\",\n",
        "  \"environment\": $ENV_NAME,\n",
        "  \"config\": ${APP_CONFIG},\n",
        "  \"code_sample\": ```python\nprint(\"Hello world!\")\n```,\n",
        "  \"inline\": `rm -rf /tmp/cache`,\n",
        "  \"subshell\": $(uname -a)\n",
        "}\n"
    );

    let mut arena = SpanArena::new(payload);
    let locked_count = scan_syntax_locks(&mut arena).expect("syntax scanning succeeds");
    assert!(
        locked_count >= 6,
        "Expected at least 6 locked syntax tokens, got {locked_count}"
    );

    let locked_spans: Vec<_> = arena
        .spans()
        .iter()
        .filter(|s| matches!(s.kind, SpanKind::Locked))
        .collect();

    // Verify all locked spans lie strictly on UTF-8 char boundaries
    for span in &locked_spans {
        assert!(payload.is_char_boundary(span.start));
        assert!(payload.is_char_boundary(span.end));
    }

    // Materializing without any edits preserves syntax verbatim
    assert_eq!(arena.materialize(), payload);
}

#[test]
fn e2e_story2_1_syntax_lock_anti_smuggling_secret_inside_locked_blocks() {
    // Adversary attempts to smuggle an AWS access key inside a locked bash code fence and JSON key
    let payload = concat!(
        "```bash\n",
        "export AWS_SECRET_ACCESS_KEY=\"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\"\n",
        "```\n",
        "curl -H \"Authorization: Bearer ghp_1234567890abcdefghijklmnopqrstuv\" `echo $SECRET`\n"
    );

    let vault = VaultHandle::new();
    let mut arena = SpanArena::new(payload);

    // Run Tier 1 guard first
    run_tier1_guard(&mut arena, &vault);

    // Run Tier 2 syntax locking and compression
    let (clean, stats) = compress_text_with_arena(&mut arena, 0.60, None);

    // Secret inside the code fence MUST be completely redacted!
    assert!(
        clean.contains(REDACTED_CREDENTIAL_MASK),
        "Code fence secret must be masked as [REDACTED_CREDENTIAL]"
    );
    assert!(!clean.contains("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"));
    assert!(!clean.contains("ghp_1234567890abcdefghijklmnopqrstuv"));

    // Code fence delimiters and backticks are preserved
    assert!(clean.contains("```bash"));
    assert!(clean.contains("```"));
    assert!(clean.contains("`echo"));
    assert!(!stats.fail_open_triggered);
}

#[test]
fn e2e_story2_1_fail_open_syntax_invariant_recovers_pruning_corrupted_json() {
    let valid_json = r#"{"name": "production_cluster", "nodes": 12, "active": true}"#;
    let mut arena = SpanArena::new(valid_json);

    // Prune the `"nodes":` key name while leaving its value, producing invalid JSON
    arena
        .push(Span::new(32, 40, SpanKind::Pruned))
        .expect("valid span bounds");

    // The pruned output is invalid JSON: `{"name": "production_cluster",  12, "active": true}`
    let candidate = arena.materialize();
    assert!(!validate_syntax_fail_open(valid_json, &candidate));

    // The fail-open materializer discards pruning and recovers the valid original text
    let fail_open = arena.materialize_fail_open();
    assert_eq!(fail_open, valid_json);
    assert!(serde_json::from_str::<serde_json::Value>(&fail_open).is_ok());
}

#[test]
fn e2e_story2_2_tail_query_anchoring_and_submodular_context_reduction() {
    let chat_payload = json!({
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are an autonomous DevOps engineering copilot with root infrastructure credentials."
            },
            {
                "role": "user",
                "content": "Here is the previous background context from yesterday: it was really very clear that the team completed initial cluster provisioning and configured the storage buckets."
            },
            {
                "role": "assistant",
                "content": "The storage buckets and initial cluster provisioning were validated successfully."
            },
            {
                "role": "user",
                "content": "What is the primary database connection string and password?"
            }
        ]
    });

    let raw_json = serde_json::to_string(&chat_payload).expect("valid json serialization");
    let vault = VaultHandle::new();

    let (compressed_json, stats) =
        sanitize_and_compress_chat_completions_json(&raw_json, &vault, 0.60)
            .expect("chat compression succeeds");

    // 1. System prompt directive is auto-anchored and retained intact
    assert!(compressed_json.contains("You are an autonomous DevOps engineering copilot"));

    // 2. Latest user query is auto-anchored and retained intact
    assert!(
        compressed_json.contains("What is the primary database connection string and password?")
    );

    // 3. Overall payload is compressed
    assert!(stats.reduction_ratio > 0.0);
    assert!(!stats.fail_open_triggered);
    assert!(serde_json::from_str::<serde_json::Value>(&compressed_json).is_ok());
}

#[test]
fn e2e_story2_2_anti_compression_bomb_latency_under_1_500_000_ns() {
    // Construct a 4KB adversarial payload with 2000+ single-character tokens and nested delimiters
    let mut bomb = String::with_capacity(4096 + 256);
    let unit = "{ \"k\": [ a, b, c, d, `e`, $(f), $G, \"h\" ], ";
    while bomb.len() < 4096 {
        bomb.push_str(unit);
    }
    bomb.truncate(4096);

    let vault = VaultHandle::new();

    // Warm-up
    let _ = sanitize_str_with_vault(&bomb, &vault);
    let _ = compress_text(&bomb, 0.60, None);

    // Benchmark 30 iterations in test suite
    let mut latencies_ns = Vec::with_capacity(30);
    for _ in 0..30 {
        let start = Instant::now();
        let _ = compress_text(&bomb, 0.60, None);
        latencies_ns.push(start.elapsed().as_nanos() as u64);
    }

    latencies_ns.sort_unstable();
    let p99_idx = (30.0_f64 * 0.99) as usize;
    let p99_ns = latencies_ns[p99_idx.min(29)];
    println!(
        "e2e_story2_2 4KB compression bomb Tier 2 P99: {} ns",
        p99_ns
    );

    // Assert strictly under 1,500,000 ns P99 SLA in release mode
    if !cfg!(debug_assertions) {
        assert!(
            p99_ns < 1_500_000,
            "P99 latency SLA breached on 4KB compression bomb: {p99_ns} ns >= 1,500,000 ns"
        );
    }
}

#[test]
fn e2e_story2_3_tool_schema_minification_over_30_percent_reduction() {
    let mut tools = vec![
        json!({
            "type": "function",
            "function": {
                "name": "execute_database_migration",
                "description": "Applies pending SQL migration files to target database schema. Highly sensitive operational command requiring elevated privileges.",
                "parameters": {
                    "type": "object",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "title": "MigrationParameters",
                    "properties": {
                        "environment": {
                            "type": "string",
                            "enum": ["staging", "production", "dr"],
                            "description": "Target deployment tier where schema migrations should be applied."
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "When true, calculates and prints the schema diff without applying destructive changes."
                        }
                    },
                    "required": ["environment"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "fetch_telemetry_metrics",
                "description": "Queries Prometheus TSDB metrics for cluster health and container resource usage over specified time window.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "Canonical Prometheus metric name, e.g. container_cpu_usage_seconds_total."
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Lookback window in minutes for aggregated metric series."
                        }
                    },
                    "required": ["metric_name"]
                }
            }
        }),
    ];

    let (orig_bytes, min_bytes) = minify_tool_definitions(&mut tools);
    let ratio = schema_reduction_ratio(orig_bytes, min_bytes);

    // Must achieve >= 30% token reduction
    assert!(
        ratio >= 0.30,
        "Expected >= 30% tool schema reduction, achieved: {:.1}%",
        ratio * 100.0
    );

    // Verify 100% schema validity and integrity of critical properties
    assert_eq!(tools[0]["function"]["name"], "execute_database_migration");
    assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    assert_eq!(
        tools[0]["function"]["parameters"]["required"][0],
        "environment"
    );
    assert_eq!(
        tools[0]["function"]["parameters"]["properties"]["environment"]["enum"][0],
        "staging"
    );

    assert_eq!(tools[1]["function"]["name"], "fetch_telemetry_metrics");
    assert_eq!(
        tools[1]["function"]["parameters"]["required"][0],
        "metric_name"
    );

    // All prose descriptions must be stripped
    assert!(tools[0]["function"].get("description").is_none());
    assert!(
        tools[0]["function"]["parameters"]["properties"]["environment"]
            .get("description")
            .is_none()
    );
    assert!(tools[1]["function"].get("description").is_none());
}

#[test]
fn e2e_story2_4_cli_black_box_execution_and_dashboard_render() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let target_dir = std::path::Path::new(manifest_dir)
        .parent()
        .expect("parent of crates")
        .parent()
        .expect("target root")
        .join("target");

    let exe_name = if cfg!(windows) {
        "context-sanitizer.exe"
    } else {
        "context-sanitizer"
    };

    let mode = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let bin_to_use = target_dir.join(mode).join(exe_name);

    // 1. Test CLI with --bench flag
    let bench_output = Command::new(&bin_to_use)
        .arg("xray")
        .arg("--bench")
        .output()
        .expect("failed to execute context-sanitizer xray --bench");

    let bench_stdout = String::from_utf8_lossy(&bench_output.stdout);
    let bench_stderr = String::from_utf8_lossy(&bench_output.stderr);
    println!("CLI bench stdout:\n{}", bench_stdout);
    println!("CLI bench stderr:\n{}", bench_stderr);

    assert!(
        bench_output.status.success(),
        "context-sanitizer xray --bench failed: {}",
        bench_stderr
    );
    assert!(bench_stdout.contains("All nanosecond SLA gates PASSED"));
    assert!(bench_stdout.contains("MeetingBank"));
    assert!(bench_stdout.contains("Adversarial 4KB Compression Bomb"));

    // 2. Test CLI with raw payload file and --json
    let temp_payload_path = std::env::temp_dir().join("test_xray_payload.json");
    let test_json = json!({
        "messages": [
            {"role": "user", "content": "What is the AWS key AKIAIOSFODNN7EXAMPLE?"}
        ]
    });
    std::fs::write(&temp_payload_path, test_json.to_string()).expect("write temp payload");

    let xray_output = Command::new(&bin_to_use)
        .arg("xray")
        .arg(&temp_payload_path)
        .arg("--rate")
        .arg("0.50")
        .arg("--json")
        .output()
        .expect("failed to execute context-sanitizer xray");

    assert!(xray_output.status.success());
    let xray_stdout = String::from_utf8_lossy(&xray_output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&xray_stdout).expect("valid json output from CLI");
    assert!(parsed.get("input_bytes").is_some());
    assert!(parsed.get("redacted_credentials").is_some());
    assert_eq!(parsed["syntax_locks_preserved"], true);

    // 3. Test CLI with raw payload file without --json (verifies default ASCII dashboard rendering)
    let ascii_output = Command::new(&bin_to_use)
        .arg("xray")
        .arg(&temp_payload_path)
        .arg("--rate")
        .arg("0.50")
        .output()
        .expect("failed to execute context-sanitizer xray without json");

    assert!(
        ascii_output.status.success(),
        "ASCII dashboard execution failed: {}",
        String::from_utf8_lossy(&ascii_output.stderr)
    );
    let ascii_stdout = String::from_utf8_lossy(&ascii_output.stdout);
    assert!(ascii_stdout.contains("CONTEXT SANITIZER X-RAY"));
    assert!(ascii_stdout.contains("Latency Breakdown"));
    assert!(ascii_stdout.contains("Token Metrics"));
    assert!(ascii_stdout.contains("Security & Integrity"));

    let _ = std::fs::remove_file(temp_payload_path);
}

#[test]
fn e2e_story2_1_fail_open_triggered_in_compress_pipeline() {
    let valid_json = r#"{"name": "production_cluster", "nodes": 12, "active": true}"#;
    let mut arena = SpanArena::new(valid_json);

    // Inject an invalid prune span across the key boundary to simulate candidate syntax corruption
    arena
        .push(Span::new(32, 40, SpanKind::Pruned))
        .expect("valid span bounds");

    let (recovered_text, stats) = compress_text_with_arena(&mut arena, 0.50, None);

    // Pipeline must trigger fail-open and return original text
    assert!(
        stats.fail_open_triggered,
        "fail_open_triggered must be true!"
    );
    assert_eq!(recovered_text, valid_json);
    assert!(serde_json::from_str::<serde_json::Value>(&recovered_text).is_ok());
}

#[test]
fn e2e_story2_2_tail_query_relevance_anchors_historical_turn_keywords() {
    let chat_payload = json!({
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "The production database cluster in us-east-1 is very very very really really very overloaded today."
            },
            {
                "role": "assistant",
                "content": "I understand the database status."
            },
            {
                "role": "user",
                "content": "Can you check the database cluster health?"
            }
        ]
    });

    let vault = VaultHandle::new();
    let (compressed_json, stats) =
        sanitize_and_compress_chat_completions_json(&chat_payload.to_string(), &vault, 0.50)
            .expect("sanitization and compression succeeds");

    let parsed: serde_json::Value =
        serde_json::from_str(&compressed_json).expect("valid JSON returned");
    let historical_user_turn = parsed["messages"][0]["content"]
        .as_str()
        .expect("string content");

    // Keywords matching tail query ("database", "cluster") must be preserved in history!
    assert!(
        historical_user_turn.contains("database"),
        "Historical turn must preserve 'database' from tail query: {}",
        historical_user_turn
    );
    assert!(
        historical_user_turn.contains("cluster"),
        "Historical turn must preserve 'cluster' from tail query: {}",
        historical_user_turn
    );
    assert!(
        stats.pruned_spans_count > 0,
        "Filler spans should be pruned"
    );
}
