#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! # `context-sanitizer`
//!
//! Pure-Rust **System One semantic firewall**, **Split-Secret Shadow Vault**, and
//! **syntax-safe token compressor** for Agentic AI, Copilot UIs, and OpenAI-compatible boundaries.
//!
//! ## Architecture Invariants
//! - **AD-1 (Span-Annotated Pipes-and-Filters Cascade):** Zero-copy, read-only byte span passes
//!   where `Redacted` and `Shadow` spans hold strict priority over `Locked`, `Pruned`, and `Retained` spans.
//! - **AD-2 (Pure-Rust Core Isolation):** `#![forbid(unsafe_code)]` with zero external C/C++ dependencies.
//! - **AD-3 (Session-Scoped Thread-Safe Vault):** Cryptographically isolated `<SHADOW:<TYPE>_<HEX6>>` tokens
//!   with destination-whitelisted `.rehydrate(args, destination)`.
//! - **AD-4 (Fail-Open Syntax Verification):** Immutable lexical locking of JSON/code delimiters with
//!   fail-open fallback guaranteeing `0%` downstream JSON parse errors.

pub mod arena;
pub mod compress;
pub mod guard;
pub mod ml;
pub mod wire;

use thiserror::Error;

pub use arena::{Span, SpanArena, SpanKind};
pub use compress::{compress_text, compress_text_with_arena, CompressionStats};
pub use guard::{
    run_tier1_guard, sanitize_str_with_vault, PiiKind, VaultHandle, REDACTED_CREDENTIAL_MASK,
};
#[cfg(feature = "ml")]
pub use ml::CandleGuard;
pub use ml::{
    chunk_and_rank_windows, detect_markdown_exfiltration, scan_fast_breaker, GatekeeperResult,
    SemanticGuard, TextWindow, ThreatGatekeeper, ThreatVerdict, Tier2Compressor, TokenCompressor,
};
pub use wire::{
    sanitize_and_compress_chat_completions_json, sanitize_chat_completions_json,
    sanitize_completions_json, ChatCompletionRequest, ChatContent, ChatMessage, CompletionPrompt,
    CompletionRequest,
};

/// High-level function to sanitize and statistically compress an OpenAI chat JSON payload.
pub fn compress_chat_json(
    payload_json: &str,
    vault: &VaultHandle,
    target_rate: f32,
) -> Result<(String, CompressionStats), SanitizerError> {
    sanitize_and_compress_chat_completions_json(payload_json, vault, target_rate)
}

/// Unified error type across the `context-sanitizer` pipeline (`AD-2`).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SanitizerError {
    /// Raised when a span's byte offsets are out of bounds or split a multi-byte UTF-8 character (`AD-1`).
    #[error("invalid span bounds {start}..{end} for buffer of length {len}")]
    InvalidSpanBounds {
        /// Inclusive start byte offset.
        start: usize,
        /// Exclusive end byte offset.
        end: usize,
        /// Total byte length of the source buffer.
        len: usize,
    },

    /// Raised when a syntax lock invariant is violated during token pruning (`AD-4`).
    #[error("syntax lock violation: {0}")]
    SyntaxLockViolation(String),

    /// Raised when `.rehydrate()` is invoked for an unregistered or external destination (`FR-3`).
    #[error("unauthorized rehydration destination: {0}")]
    UnauthorizedRehydration(String),

    /// Raised when an OpenAI wire payload cannot be parsed or serialized (`FR-10`).
    #[error("wire payload parse error: {0}")]
    WireParseError(String),

    /// Raised when the System One threat gatekeeper blocks a prompt injection or exfiltration attempt (`FR-8`).
    #[error("threat detected (score: {score_bps} bps): {reason}")]
    ThreatBlocked {
        /// Threat score in basis points (0..=10000).
        score_bps: u16,
        /// Human-readable reason for the block.
        reason: String,
    },

    /// Raised when a neural model checkpoint or tokenizer fails to load (`AD-5`).
    #[error("model load error: {0}")]
    ModelLoadError(String),

    /// Raised when guard evaluation exceeds the nanosecond watchdog deadline (`NFR-1`).
    #[error("guard evaluation timeout: {0} ns")]
    GuardTimeout(u64),
}

/// Sanitizes a raw text payload using a newly created session [`VaultHandle`] pre-configured with
/// `allowed_destinations`, returning `(sanitized_text, vault_handle)`.
#[must_use]
pub fn sanitize_text<I, S>(input: &str, allowed_destinations: I) -> (String, VaultHandle)
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let vault = VaultHandle::with_destinations(allowed_destinations);
    let sanitized = sanitize_str_with_vault(input, &vault);
    (sanitized, vault)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_text_round_trip_works() {
        let (clean, vault) = sanitize_text(
            "User alice@acme.com leaked AKIAIOSFODNN7EXAMPLE",
            ["crm_lookup"],
        );
        assert!(clean.contains("[REDACTED_CREDENTIAL]"));
        assert!(!clean.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(clean.contains("<SHADOW:EMAIL_"));
        let rehydrated = vault.rehydrate(&clean, "crm_lookup");
        assert_eq!(
            rehydrated,
            Ok("User alice@acme.com leaked [REDACTED_CREDENTIAL]".to_owned())
        );
    }
}
