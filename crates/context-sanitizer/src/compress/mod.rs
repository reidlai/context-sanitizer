//! Tier 2 (`L0.5`) zero-weight syntax-safe statistical context compressor (`FR-4`, `FR-5`, `FR-6`, `AD-4`).
//!
//! Integrates pure-Rust lexical syntax locking (`logos`), static Shannon unigram surprisal scoring,
//! tail-query anchored Okapi BM25 relevance, monotone submodular $(1 - 1/e)$ knapsack selection,
//! OpenAI JSON tool schema minification, and fail-open syntax verification.

pub mod bm25;
pub mod lexer;
pub mod schema;
pub mod shannon;
pub mod submodular;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::arena::SpanArena;
pub use schema::minify_tool_definitions;

/// Diagnostic telemetry and performance metrics for a Tier 2 compression execution.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Byte length of input payload prior to compression.
    pub original_bytes: usize,
    /// Byte length of output payload after compression and fail-open verification.
    pub compressed_bytes: usize,
    /// Estimated token count of original text ($\sim 4$ bytes/token heuristic).
    pub original_tokens: usize,
    /// Estimated token count of compressed output.
    pub compressed_tokens: usize,
    /// Overall token reduction ratio ($0.0 \dots 1.0$).
    pub reduction_ratio: f32,
    /// Count of structural syntax tokens locked against pruning.
    pub syntax_locked_count: usize,
    /// Count of low-significance token spans removed.
    pub pruned_spans_count: usize,
    /// `true` if the Fail-Open Invariant was triggered to restore pruned content due to syntax corruption.
    pub fail_open_triggered: bool,
    /// Elapsed wall-clock nanoseconds spent in Tier 2 compression passes.
    pub elapsed_ns: u64,
}

/// Compresses a raw text payload using Tier 2 statistical pruning with lexical syntax locking
/// and fail-open verification.
#[must_use]
pub fn compress_text(
    input: &str,
    target_rate: f32,
    tail_query: Option<&str>,
) -> (String, CompressionStats) {
    let mut arena = SpanArena::with_capacity(input, (input.len() / 4).max(16));
    compress_text_with_arena(&mut arena, target_rate, tail_query)
}

/// Executes Tier 2 compression passes on an existing [`SpanArena`] (e.g. after Tier 1 guard passes).
#[must_use]
pub fn compress_text_with_arena(
    arena: &mut SpanArena<'_>,
    target_rate: f32,
    tail_query: Option<&str>,
) -> (String, CompressionStats) {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let start = Instant::now();
    let original_bytes = arena.source().len();
    let original_tokens = original_bytes.div_ceil(4);

    if original_bytes == 0 {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        let elapsed_ns = 0u64;

        return (
            String::new(),
            CompressionStats {
                original_bytes: 0,
                compressed_bytes: 0,
                original_tokens: 0,
                compressed_tokens: 0,
                reduction_ratio: 0.0,
                syntax_locked_count: 0,
                pruned_spans_count: 0,
                fail_open_triggered: false,
                elapsed_ns,
            },
        );
    }

    // 1. Scan and lock structural syntax tokens (JSON delimiters, keys, fences, backticks, $VAR)
    let syntax_locked_count = lexer::scan_syntax_locks(arena).unwrap_or(0);

    // 2. Perform lazy-greedy knapsack statistical pruning
    let pruned_spans_count =
        submodular::prune_unlocked_context(arena, target_rate, tail_query).unwrap_or(0);

    // 3. Attempt materialization with pruning applied
    let candidate = arena.materialize();

    // 4. Verify syntax integrity under the Fail-Open Syntax Invariant (`AD-4`).
    // If no spans were pruned, the syntax is guaranteed unmodified by compression.
    let (final_text, fail_open_triggered) = if pruned_spans_count == 0
        || lexer::validate_syntax_fail_open(arena.source(), &candidate)
    {
        (candidate, false)
    } else {
        // Fail open: discard all prune deletions while preserving Tier 1 Redacted/Shadow spans!
        (arena.materialize_fail_open(), true)
    };

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let elapsed_ns = start.elapsed().as_nanos() as u64;
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let elapsed_ns = 0u64;

    let compressed_bytes = final_text.len();
    let compressed_tokens = compressed_bytes.div_ceil(4);
    let reduction_ratio = if original_tokens > 0 && compressed_tokens < original_tokens {
        ((original_tokens - compressed_tokens) as f32) / (original_tokens as f32)
    } else {
        0.0
    };

    (
        final_text,
        CompressionStats {
            original_bytes,
            compressed_bytes,
            original_tokens,
            compressed_tokens,
            reduction_ratio,
            syntax_locked_count,
            pruned_spans_count,
            fail_open_triggered,
            elapsed_ns,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_text_reduces_conversational_text_and_preserves_syntax() {
        let text = "Here is the code block:\n```python\ndef get_val():\n    return 42\n```\nIt is really very clear that the user requested assistance with the database configuration today.";
        let (output, stats) = compress_text(text, 0.60, Some("database configuration"));

        assert!(stats.reduction_ratio > 0.0);
        assert!(stats.syntax_locked_count > 0);
        assert!(output.contains("```python"));
        assert!(output.contains("database"));
        assert!(!stats.fail_open_triggered);
    }
}
