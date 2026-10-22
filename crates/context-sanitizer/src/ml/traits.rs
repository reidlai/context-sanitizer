//! Pluggable polymorphic trait boundaries for threat scoring and compression (`FR-7`, `AD-5`).

use crate::SanitizerError;

/// Thread-safe semantic threat evaluation guard (`FR-7`, `AD-5`).
///
/// Implementors score incoming text payloads for semantic threat / prompt injection risk.
/// Returns a threat score in the range `0.0..=1.0` (where `0.0` is benign and `1.0` is malicious).
pub trait SemanticGuard: Send + Sync {
    /// Evaluates the threat score of the given `text`.
    ///
    /// # Errors
    /// Returns [`SanitizerError`] if evaluation fails.
    fn evaluate_threat(&self, text: &str) -> Result<f32, SanitizerError>;
}

/// Thread-safe pluggable token compressor (`FR-7`, `AD-5`).
///
/// Allows external compression engines (e.g. LLMLingua, custom neural summarizers)
/// to be registered polymorphically into the context sanitization pipeline.
pub trait TokenCompressor: Send + Sync {
    /// Compresses `payload` aiming for `target_rate` ($0.0 \le \text{rate} \le 1.0$).
    ///
    /// # Errors
    /// Returns [`SanitizerError`] if compression fails.
    fn compress(&self, payload: &str, target_rate: f32) -> Result<String, SanitizerError>;
}

/// Default Tier 2 statistical token compressor implementing [`TokenCompressor`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Tier2Compressor;

impl TokenCompressor for Tier2Compressor {
    fn compress(&self, payload: &str, target_rate: f32) -> Result<String, SanitizerError> {
        let (compressed, _) = crate::compress::compress_text(payload, target_rate, None);
        Ok(compressed)
    }
}
