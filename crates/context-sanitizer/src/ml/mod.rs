//! Tier 3 Pluggable System One ML Threat Gatekeeper & Fault-Isolated Fallback (`FR-7`, `FR-8`, `FR-9`, `AD-5`).
//!
//! Orchestrates Stage 3A-Fast SIMD lexical pattern interception, surprisal-ranked Stage 3B-Neural
//! sliding-window inference, execution deadline watchdogs (`guard_timeout_ns`), and zero-panic
//! graceful fallback to Tier 2 statistical compression on missing, corrupted, or timed-out weights.

#[cfg(feature = "ml")]
pub mod candle;
pub mod chunking;
pub mod fast_breaker;
pub mod traits;

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::arena::SpanArena;
use crate::guard::run_tier1_guard;
use crate::SanitizerError;
use crate::VaultHandle;

pub use chunking::{
    chunk_and_rank_windows, extract_token_spans, TextWindow, DEFAULT_MAX_WINDOWS_K,
    DEFAULT_WINDOW_OVERLAP, DEFAULT_WINDOW_SIZE,
};
pub use fast_breaker::{detect_markdown_exfiltration, scan_fast_breaker, FAST_BREAKER_THREAT_BPS};
pub use traits::{SemanticGuard, Tier2Compressor, TokenCompressor};

#[cfg(feature = "ml")]
pub use candle::CandleGuard;

/// Default threat sensitivity threshold (0.85). Scores $\ge$ threshold are blocked.
pub const DEFAULT_THREAT_THRESHOLD: f32 = 0.85;

/// Default hard nanosecond watchdog deadline (5,000,000 ns / 5 ms P99 ceiling).
pub const DEFAULT_GUARD_TIMEOUT_NS: u64 = 5_000_000;

/// Final verdict produced by the Threat Gatekeeper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThreatVerdict {
    /// Payload passed all security gates without crossing threat thresholds.
    Pass,
    /// Payload was intercepted and blocked due to structural injection or high threat score.
    Blocked {
        /// Threat score in basis points (0..=10000).
        score_bps: u16,
        /// Reason for interception.
        reason: String,
    },
    /// Neural inference was skipped or faulted; gracefully fell back to Tier 2 compression.
    Fallback {
        /// Reason for fallback (timeout, model load error, or missing checkpoint).
        reason: String,
    },
}

impl ThreatVerdict {
    /// Returns `true` if the verdict allows the payload to proceed.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        matches!(self, ThreatVerdict::Pass | ThreatVerdict::Fallback { .. })
    }

    /// Returns `true` if the payload was blocked.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        matches!(self, ThreatVerdict::Blocked { .. })
    }
}

/// Telemetry and diagnostics produced by a Threat Gatekeeper execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatekeeperResult {
    /// Threat gatekeeper verdict.
    pub verdict: ThreatVerdict,
    /// Maximum observed continuous threat score (0.0 .. 1.0).
    pub max_threat_score: f32,
    /// Number of candidate sliding windows evaluated by neural model.
    pub evaluated_windows: usize,
    /// Total elapsed wall-clock nanoseconds in Tier 3 processing.
    pub latency_ns: u64,
    /// Whether fallback to Tier 2 statistical compression was triggered.
    pub fallback_triggered: bool,
}

/// Pluggable Tier 3 Threat Gatekeeper and pipeline coordinator (`FR-7`, `FR-8`, `FR-9`, `AD-5`).
pub struct ThreatGatekeeper {
    guard: Option<Arc<dyn SemanticGuard>>,
    compressor: Arc<dyn TokenCompressor>,
    threat_threshold: f32,
    guard_timeout_ns: u64,
    window_size: usize,
    window_overlap: usize,
    max_windows_k: usize,
}

impl Default for ThreatGatekeeper {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreatGatekeeper {
    /// Constructs a default [`ThreatGatekeeper`] with Stage 3A-Fast lexical breaker,
    /// default Tier 2 statistical compression, and no neural model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            guard: None,
            compressor: Arc::new(Tier2Compressor),
            threat_threshold: DEFAULT_THREAT_THRESHOLD,
            guard_timeout_ns: DEFAULT_GUARD_TIMEOUT_NS,
            window_size: DEFAULT_WINDOW_SIZE,
            window_overlap: DEFAULT_WINDOW_OVERLAP,
            max_windows_k: DEFAULT_MAX_WINDOWS_K,
        }
    }

    /// Configures a custom [`SemanticGuard`] implementation.
    #[must_use]
    pub fn with_guard<G: SemanticGuard + 'static>(mut self, guard: G) -> Self {
        self.guard = Some(Arc::new(guard));
        self
    }

    /// Configures an `Arc`-wrapped [`SemanticGuard`] implementation.
    #[must_use]
    pub fn with_guard_arc(mut self, guard: Arc<dyn SemanticGuard>) -> Self {
        self.guard = Some(guard);
        self
    }

    /// Configures a custom [`TokenCompressor`] implementation.
    #[must_use]
    pub fn with_compressor<C: TokenCompressor + 'static>(mut self, compressor: C) -> Self {
        self.compressor = Arc::new(compressor);
        self
    }

    /// Sets the threat sensitivity threshold ($0.0 \dots 1.0$).
    #[must_use]
    pub fn with_threat_threshold(mut self, threshold: f32) -> Self {
        self.threat_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets the nanosecond watchdog execution ceiling (`guard_timeout_ns`).
    #[must_use]
    pub fn with_guard_timeout_ns(mut self, timeout_ns: u64) -> Self {
        self.guard_timeout_ns = timeout_ns;
        self
    }

    /// Sets the maximum number of sliding windows to evaluate ($K \le 2$).
    #[must_use]
    pub fn with_max_windows_k(mut self, max_k: usize) -> Self {
        self.max_windows_k = max_k.clamp(1, 10);
        self
    }

    /// Sets custom sliding window size and overlap parameters in tokens.
    #[must_use]
    pub fn with_window_params(mut self, size: usize, overlap: usize) -> Self {
        self.window_size = size.max(1);
        self.window_overlap = overlap.min(self.window_size.saturating_sub(1));
        self
    }

    /// Returns the configured threat sensitivity threshold.
    #[must_use]
    pub fn threat_threshold(&self) -> f32 {
        self.threat_threshold
    }

    /// Evaluates threat on raw `text` through Stage 3A-Fast breaker and Stage 3B-Neural sliding windows.
    ///
    /// # Errors
    /// Returns [`SanitizerError::ThreatBlocked`] if Stage 3A detects injection patterns or if any
    /// window crosses the threat threshold.
    pub fn evaluate_threat(&self, text: &str) -> Result<GatekeeperResult, SanitizerError> {
        let start = Instant::now();

        // 1. Stage 3A-Fast SIMD Lexical Breaker (< 100,000 ns)
        scan_fast_breaker(text)?;

        // 2. Stage 3B-Neural Sliding Windows (if guard configured)
        let mut max_threat_score = 0.0f32;
        let mut evaluated_windows = 0usize;
        let mut fallback_triggered = false;
        let mut fallback_reason = None;

        if let Some(ref guard) = self.guard {
            let windows = chunk_and_rank_windows(
                text,
                self.window_size,
                self.window_overlap,
                self.max_windows_k,
            );

            let guard_start = Instant::now();

            for window in windows {
                evaluated_windows += 1;
                let eval_res = guard.evaluate_threat(window.text);
                let elapsed_ns = guard_start.elapsed().as_nanos() as u64;

                match eval_res {
                    Ok(score) => {
                        max_threat_score = max_threat_score.max(score);
                        if score >= self.threat_threshold {
                            let score_bps = (score * 10_000.0).round().clamp(0.0, 10_000.0) as u16;
                            return Err(SanitizerError::ThreatBlocked {
                                score_bps,
                                reason: format!(
                                    "Stage 3B threat score {:.4} exceeded threshold {:.4} in window {}",
                                    score, self.threat_threshold, window.index
                                ),
                            });
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "[WARN][context-sanitizer::ml] Neural evaluation error ({err}); falling back gracefully"
                        );
                        fallback_triggered = true;
                        fallback_reason = Some(err.to_string());
                        break;
                    }
                }

                if elapsed_ns >= self.guard_timeout_ns {
                    eprintln!(
                        "[WARN][context-sanitizer::ml] Guard watchdog timeout ({elapsed_ns} ns >= {} ns ceiling); falling back gracefully",
                        self.guard_timeout_ns
                    );
                    fallback_triggered = true;
                    fallback_reason = Some(format!(
                        "guard timeout ({elapsed_ns} ns >= {} ns ceiling)",
                        self.guard_timeout_ns
                    ));
                    break;
                }
            }
        }

        let latency_ns = start.elapsed().as_nanos() as u64;
        let verdict = if let Some(reason) = fallback_reason {
            ThreatVerdict::Fallback { reason }
        } else {
            ThreatVerdict::Pass
        };

        Ok(GatekeeperResult {
            verdict,
            max_threat_score,
            evaluated_windows,
            latency_ns,
            fallback_triggered,
        })
    }

    /// Full end-to-end pipeline: Tier 1 secrets/PII protection + Stage 3A fast-breaker
    /// + Stage 3B neural sliding windows + Token compressor (with Tier 2 fallback).
    ///
    /// # Errors
    /// Returns [`SanitizerError::ThreatBlocked`] if structural injection is detected or threat
    /// threshold is crossed.
    pub fn sanitize_and_compress(
        &self,
        input: &str,
        target_rate: f32,
        vault: Option<&VaultHandle>,
    ) -> Result<(String, GatekeeperResult), SanitizerError> {
        let t3_start = Instant::now();

        // 1. Tier 1 credential redaction & PII vaulting
        let tier1_text = if let Some(v) = vault {
            let mut arena = SpanArena::new(input);
            run_tier1_guard(&mut arena, v);
            arena.materialize_fail_open()
        } else {
            let local_vault = VaultHandle::new();
            let mut arena = SpanArena::new(input);
            run_tier1_guard(&mut arena, &local_vault);
            arena.materialize_fail_open()
        };

        // 2. Stage 3A-Fast Lexical Breaker (< 100,000 ns)
        scan_fast_breaker(&tier1_text)?;

        // 3. Stage 3B-Neural Inference
        let mut max_threat_score = 0.0f32;
        let mut evaluated_windows = 0usize;
        let mut fallback_triggered = false;
        let mut fallback_reason = None;

        if let Some(ref guard) = self.guard {
            let windows = chunk_and_rank_windows(
                &tier1_text,
                self.window_size,
                self.window_overlap,
                self.max_windows_k,
            );

            let guard_start = Instant::now();

            for window in windows {
                evaluated_windows += 1;
                let eval_res = guard.evaluate_threat(window.text);
                let elapsed_ns = guard_start.elapsed().as_nanos() as u64;

                match eval_res {
                    Ok(score) => {
                        max_threat_score = max_threat_score.max(score);
                        if score >= self.threat_threshold {
                            let score_bps = (score * 10_000.0).round().clamp(0.0, 10_000.0) as u16;
                            return Err(SanitizerError::ThreatBlocked {
                                score_bps,
                                reason: format!(
                                    "Stage 3B threat score {:.4} exceeded threshold {:.4} in window {}",
                                    score, self.threat_threshold, window.index
                                ),
                            });
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "[WARN][context-sanitizer::ml] Neural evaluation error ({err}); falling back to Tier 2 statistical compression"
                        );
                        fallback_triggered = true;
                        fallback_reason = Some(err.to_string());
                        break;
                    }
                }

                if elapsed_ns >= self.guard_timeout_ns {
                    eprintln!(
                        "[WARN][context-sanitizer::ml] Guard watchdog timeout ({elapsed_ns} ns >= {} ns ceiling); falling back to Tier 2 statistical compression",
                        self.guard_timeout_ns
                    );
                    fallback_triggered = true;
                    fallback_reason = Some(format!(
                        "guard timeout ({elapsed_ns} ns >= {} ns ceiling)",
                        self.guard_timeout_ns
                    ));
                    break;
                }
            }
        }

        // 4. Token Compression pass
        let compressed = match self.compressor.compress(&tier1_text, target_rate) {
            Ok(c) => c,
            Err(err) => {
                eprintln!(
                    "[WARN][context-sanitizer::ml] Primary compressor error ({err}); falling back to default Tier 2 compression"
                );
                fallback_triggered = true;
                if fallback_reason.is_none() {
                    fallback_reason = Some(err.to_string());
                }
                let (c, _) = crate::compress::compress_text(&tier1_text, target_rate, None);
                c
            }
        };

        let latency_ns = t3_start.elapsed().as_nanos() as u64;
        let verdict = if let Some(reason) = fallback_reason {
            ThreatVerdict::Fallback { reason }
        } else {
            ThreatVerdict::Pass
        };

        Ok((
            compressed,
            GatekeeperResult {
                verdict,
                max_threat_score,
                evaluated_windows,
                latency_ns,
                fallback_triggered,
            },
        ))
    }
}
