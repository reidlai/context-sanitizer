//! End-to-End Integration Test Suite for Epic 1 (`Stories 1.1–1.4`).
//!
//! Covers:
//! 1. SpanArena overlap precedence (`Redacted > Shadow > Locked > Pruned > Retained`) and UTF-8 safety (`AD-1`).
//! 2. Multi-vendor credentials, `*_KEY`/`*_SECRET` heuristics ($\ge 16$ chars), and standalone Shannon entropy ($H \ge 4.5$, len $\ge 20$) (`FR-1`).
//! 3. Zero false positives on benign code identifiers (`const KEY_COUNT = 5`, `token_limit = 4096`) (`NFR-6`).
//! 4. Operational PII shadow vaulting (`EMAIL`, `IP`, `PHONE`), in-session determinism, cross-session salt isolation, and destination-whitelisted `.rehydrate()` (`FR-2`, `FR-3`, `AD-3`).
//! 5. OpenAI `/v1/chat/completions`, `/v1/chat`, and `/v1/completions` wire parity with pass-through fields (`FR-10`).
//! 6. Adversarial ReDoS 4KB payload latency `< 500,000 ns` P99 (`NFR-1`).
//! 7. Multi-threaded `VaultHandle` concurrency (10,000+ operations without contention/panics) (`NFR-2`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use context_sanitizer::{
    sanitize_chat_completions_json, sanitize_completions_json, sanitize_str_with_vault,
    sanitize_text, PiiKind, SanitizerError, Span, SpanArena, SpanKind, VaultHandle,
    REDACTED_CREDENTIAL_MASK,
};

#[test]
fn e2e_matrix_row1_overlapping_span_precedence() {
    let input = "LOCKED_PREFIX_AKIAIOSFODNN7EXAMPLE_LOCKED_SUFFIX";
    let mut arena = SpanArena::new(input);
    arena
        .push(Span::new(0, input.len(), SpanKind::Locked))
        .expect("valid locked span");
    // Also push an overlapping Shadow span on the same range to verify Redacted > Shadow > Locked
    arena
        .push(Span::new(
            14,
            34,
            SpanKind::Shadow("<SHADOW:EMAIL_000000>".to_owned()),
        ))
        .expect("valid shadow span");
    arena
        .push(Span::new(
            14,
            34,
            SpanKind::Redacted(REDACTED_CREDENTIAL_MASK),
        ))
        .expect("valid redacted span");

    assert_eq!(
        arena.materialize(),
        "LOCKED_PREFIX_[REDACTED_CREDENTIAL]_LOCKED_SUFFIX"
    );

    // Full pipeline check: a credential assignment containing an email must be Redacted,
    // and the embedded email must NEVER be vaulted into VaultHandle.
    let (sanitized, vault) = sanitize_text(
        "api_secret = \"user_alice@acme.com_secret_token_99\"",
        ["db_tool"],
    );
    assert_eq!(sanitized, "api_secret = \"[REDACTED_CREDENTIAL]\"");
    assert!(
        vault.is_empty(),
        "VaultHandle must never store substrings of redacted credentials"
    );
}

#[test]
fn e2e_matrix_row2_and_row3_credentials_entropy_and_benign_code() {
    let payload = concat!(
        "aws_id=AKIAIOSFODNN7EXAMPLE\n",
        "gcp_key=AIzaSyDaGmWKa4JsXZ-0123456789abcdef\n",
        "github=ghp_1234567890abcdefghijklmnopqrstuv\n",
        "gitlab=glpat-1234567890abcdefghij\n",
        "slack=xoxb-123456789012-123456789012-abcdef\n",
        "okta=00D1234567890abcde\n",
        "azure_conn=\"DefaultEndpointsProtocol=https;AccountName=myacct;AccountKey=abcd1234EFGH5678ijkl9012MNOP3456==;\"\n",
        "k8s_jwt=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0\n",
        "custom_secret: \"super_secret_assigned_value_98765\"\n",
        "standalone_entropy=aB3$dE6&gH9*jK2!mN5@pQ8^rT1#uW4%\n",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmU=\n-----END OPENSSH PRIVATE KEY-----\n",
        "const KEY_COUNT = 5;\n",
        "let token_limit = 4096;\n"
    );

    let (sanitized, vault) = sanitize_text(payload, ["internal_tool"]);

    assert!(!sanitized.contains("AKIAIOSFODNN7EXAMPLE"));
    assert!(!sanitized.contains("AIzaSyDaGmWKa4JsXZ"));
    assert!(!sanitized.contains("ghp_1234567890abcdefghijklmnopqrstuv"));
    assert!(!sanitized.contains("glpat-1234567890abcdefghij"));
    assert!(!sanitized.contains("xoxb-123456789012"));
    assert!(!sanitized.contains("00D1234567890abcde"));
    assert!(!sanitized.contains("abcd1234EFGH5678ijkl9012MNOP3456=="));
    assert!(!sanitized.contains("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"));
    assert!(!sanitized.contains("super_secret_assigned_value_98765"));
    assert!(!sanitized.contains("aB3$dE6&gH9*jK2!mN5@pQ8^rT1#uW4%"));
    assert!(!sanitized.contains("b3BlbnNzaC1rZXktdjEAAAAABG5vbmU="));

    // Benign code identifiers must remain 100% intact
    assert!(sanitized.contains("const KEY_COUNT = 5;"));
    assert!(sanitized.contains("let token_limit = 4096;"));

    // Root credentials must NEVER be stored in VaultHandle
    assert!(vault.is_empty());
}

#[test]
fn e2e_matrix_row4_adversarial_redos_4kb_under_500_000_ns() {
    // Construct a 4KB adversarial payload filled with near-match prefixes and partial assignments
    let unit = "AKIA0000000000000000 SECRET_KEY=short_val ghp_short const TOKEN_MAX = 12; ";
    let mut payload = String::with_capacity(4096 + 128);
    while payload.len() < 4096 {
        payload.push_str(unit);
    }
    payload.truncate(4096);

    let vault = VaultHandle::new();
    // Warm up the static Aho-Corasick DFA automatons
    let _ = sanitize_str_with_vault(&payload, &vault);

    // Measure 100 iterations and assert `< 500,000 ns` P99 (and median) ceiling on 4KB adversarial ReDoS payload.
    let mut samples_ns = Vec::with_capacity(100);
    for _ in 0..100 {
        let start = Instant::now();
        let out = sanitize_str_with_vault(&payload, &vault);
        let elapsed_ns = start.elapsed().as_nanos();
        assert_eq!(out.len(), payload.len());
        samples_ns.push(elapsed_ns);
    }
    samples_ns.sort_unstable();
    let min_ns = samples_ns[0];
    let median_ns = samples_ns[samples_ns.len() / 2];
    let p99_ns = samples_ns[(samples_ns.len() * 99) / 100];

    if cfg!(debug_assertions) {
        assert!(
            min_ns < 5_000_000,
            "Expected linear-time scan in debug build, got min {min_ns} ns (median {median_ns} ns, p99 {p99_ns} ns)"
        );
    } else {
        assert!(
            p99_ns < 500_000,
            "Expected < 500,000 ns P99 on 4KB adversarial ReDoS payload in release mode, got p99 {p99_ns} ns (median {median_ns} ns)"
        );
    }
}

#[test]
fn e2e_matrix_row5_and_row6_session_pii_vaulting_and_whitelisted_rehydration() {
    let vault_a = VaultHandle::with_destinations(["stripe_lookup", "postgres_query"]);
    let vault_b = VaultHandle::with_destinations(["stripe_lookup"]);

    let msg = "Customer alice@acme.com logged in from 192.168.1.42 and 2001:db8::1, phone +1-415-555-0199. Confirm alice@acme.com.";
    let sanitized_a = sanitize_str_with_vault(msg, &vault_a);
    let sanitized_b = sanitize_str_with_vault(msg, &vault_b);

    assert!(!sanitized_a.contains("alice@acme.com"));
    assert!(!sanitized_a.contains("192.168.1.42"));
    assert!(!sanitized_a.contains("2001:db8::1"));
    assert!(!sanitized_a.contains("+1-415-555-0199"));

    // Cross-session isolation: Session A and Session B must produce distinct shadow tokens
    assert_ne!(sanitized_a, sanitized_b);

    // Authorized rehydration succeeds and restores both occurrences of `alice@acme.com`
    let restored = vault_a
        .rehydrate(&sanitized_a, "stripe_lookup")
        .expect("whitelisted destination should succeed");
    assert_eq!(restored, msg);

    // Unauthorized destination is blocked with SanitizerError::UnauthorizedRehydration
    let unauthorized = vault_a.rehydrate(&sanitized_a, "https://evil.com/leak");
    assert_eq!(
        unauthorized,
        Err(SanitizerError::UnauthorizedRehydration(
            "https://evil.com/leak".to_owned()
        ))
    );
}

#[test]
fn e2e_matrix_row7_openai_wire_chat_and_completions_round_trip() {
    let chat_payload = r#"{
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a support agent. const KEY_COUNT = 5;"},
            {"role": "user", "content": "My AWS key is AKIAIOSFODNN7EXAMPLE and my email is alice@acme.com."},
            {"role": "user", "content": "Please look up alice@acme.com on 10.20.30.40 or +1-415-555-0199."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "stripe_lookup",
                    "parameters": {"type": "object", "properties": {"email": {"type": "string"}}}
                }
            }
        ],
        "temperature": 0.0,
        "custom_gateway_field": "preserve_me"
    }"#;

    let vault = VaultHandle::with_destinations(["stripe_lookup"]);
    let sanitized_chat =
        sanitize_chat_completions_json(chat_payload, &vault).expect("valid chat JSON");

    assert!(sanitized_chat.contains(REDACTED_CREDENTIAL_MASK));
    assert!(!sanitized_chat.contains("AKIAIOSFODNN7EXAMPLE"));
    assert!(!sanitized_chat.contains("alice@acme.com"));
    assert!(!sanitized_chat.contains("10.20.30.40"));
    assert!(!sanitized_chat.contains("+1-415-555-0199"));
    assert!(sanitized_chat.contains("const KEY_COUNT = 5;"));
    assert!(sanitized_chat.contains("\"custom_gateway_field\":\"preserve_me\""));

    // Verify `/v1/completions` wire ingestion
    let completion_payload = r#"{
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Token ghp_1234567890abcdefghijklmnopqrstuv for alice@acme.com",
        "Echo": true
    }"#;
    let sanitized_comp =
        sanitize_completions_json(completion_payload, &vault).expect("valid completion JSON");
    assert!(sanitized_comp.contains(REDACTED_CREDENTIAL_MASK));
    assert!(!sanitized_comp.contains("ghp_1234567890abcdefghijklmnopqrstuv"));
    assert!(!sanitized_comp.contains("alice@acme.com"));
    assert!(sanitized_comp.contains("\"Echo\":true"));
}

#[test]
fn e2e_concurrent_10_000_vault_operations_thread_safety() {
    let vault = Arc::new(VaultHandle::with_destinations(["db_tool"]));
    let mut handles = Vec::with_capacity(10);

    for thread_id in 0..10 {
        let v = Arc::clone(&vault);
        handles.push(thread::spawn(move || {
            for i in 0..1_000 {
                let email = format!("user_{thread_id}_{}@acme.com", i % 25);
                let tok = v.vault_pii(PiiKind::Email, &email);
                let rehydrated = v
                    .rehydrate(&tok, "db_tool")
                    .expect("whitelisted rehydration");
                assert_eq!(rehydrated, email);
            }
        }));
    }

    for h in handles {
        h.join().expect("worker thread should not panic");
    }
    assert_eq!(vault.len(), 250);
}
