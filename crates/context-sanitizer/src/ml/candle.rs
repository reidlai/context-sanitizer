//! Candle SafeTensors encoder loader and CPU neural threat scoring (`FR-8`, `FR-9`, `AD-5`).
//!
//! Loads quantized SafeTensors weights and tokenizers to run CPU sequence classification
//! on candidate surprisal windows, returning continuous threat scores ($0.0 \dots 1.0$).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::ml::traits::SemanticGuard;
use crate::SanitizerError;

/// SafeTensors CPU neural threat evaluation guard (`FR-8`, `AD-5`).
pub struct CandleGuard {
    tokenizer: Tokenizer,
    device: Device,
    tensors: HashMap<String, Tensor>,
}

impl CandleGuard {
    /// Loads a [`CandleGuard`] from SafeTensors model weights and a HuggingFace `tokenizer.json`.
    ///
    /// # Errors
    /// Returns [`SanitizerError::ModelLoadError`] if files are missing, corrupted, or incompatible.
    pub fn from_files<P1: AsRef<Path>, P2: AsRef<Path>>(
        weights_path: P1,
        tokenizer_path: P2,
    ) -> Result<Self, SanitizerError> {
        let w_path = weights_path.as_ref();
        let t_path = tokenizer_path.as_ref();

        if !w_path.exists() {
            return Err(SanitizerError::ModelLoadError(format!(
                "SafeTensors checkpoint path not found: {}",
                w_path.display()
            )));
        }

        if !t_path.exists() {
            return Err(SanitizerError::ModelLoadError(format!(
                "Tokenizer configuration not found: {}",
                t_path.display()
            )));
        }

        let tokenizer = Tokenizer::from_file(t_path).map_err(|err| {
            SanitizerError::ModelLoadError(format!(
                "Failed to deserialize tokenizer from {}: {err}",
                t_path.display()
            ))
        })?;

        let tensors = candle_core::safetensors::load(w_path, &Device::Cpu).map_err(|err| {
            SanitizerError::ModelLoadError(format!(
                "Failed to deserialize safetensors from {}: {err}",
                w_path.display()
            ))
        })?;

        Ok(Self {
            tokenizer,
            device: Device::Cpu,
            tensors,
        })
    }

    /// Evaluates the semantic threat score of `text` on CPU.
    ///
    /// # Errors
    /// Returns [`SanitizerError::ModelLoadError`] if tensor shapes mismatch or tokenization fails.
    pub fn evaluate(&self, text: &str) -> Result<f32, SanitizerError> {
        let encoding = self.tokenizer.encode(text, true).map_err(|err| {
            SanitizerError::ModelLoadError(format!("Tokenizer encoding failed: {err}"))
        })?;

        let token_ids = encoding.get_ids();
        if token_ids.is_empty() {
            return Ok(0.0);
        }

        // Bound tokens to at most 512 to prevent subword expansion from overflowing position embeddings
        let token_ids = if token_ids.len() > 512 {
            &token_ids[..512]
        } else {
            token_ids
        };

        // 1. Look up embeddings tensor
        let embeddings = self.tensors.get("embeddings.weight").ok_or_else(|| {
            SanitizerError::ModelLoadError("Missing 'embeddings.weight' tensor".to_owned())
        })?;

        // 2. Select token embeddings
        let ids_tensor = Tensor::new(token_ids, &self.device).map_err(|err| {
            SanitizerError::ModelLoadError(format!("Failed to allocate ids tensor: {err}"))
        })?;

        let token_embeds = embeddings.index_select(&ids_tensor, 0).map_err(|err| {
            SanitizerError::ModelLoadError(format!("Embedding index selection failed: {err}"))
        })?;

        // 3. Mean pooling across sequence dimension (dim 0)
        let pooled = token_embeds.mean(0).map_err(|err| {
            SanitizerError::ModelLoadError(format!("Mean pooling over embeddings failed: {err}"))
        })?;

        // 4. Classifier linear projection
        let classifier_w = self.tensors.get("classifier.weight").ok_or_else(|| {
            SanitizerError::ModelLoadError("Missing 'classifier.weight' tensor".to_owned())
        })?;

        let pooled_2d = pooled.unsqueeze(0).map_err(|err| {
            SanitizerError::ModelLoadError(format!("Reshape for matmul failed: {err}"))
        })?;

        let w_t = classifier_w.t().map_err(|err| {
            SanitizerError::ModelLoadError(format!("Classifier weights transpose failed: {err}"))
        })?;

        let mut logits = pooled_2d
            .matmul(&w_t)
            .map_err(|err| {
                SanitizerError::ModelLoadError(format!("Classifier linear matmul failed: {err}"))
            })?
            .squeeze(0)
            .map_err(|err| {
                SanitizerError::ModelLoadError(format!("Logits squeeze failed: {err}"))
            })?;

        if let Some(bias) = self.tensors.get("classifier.bias") {
            logits = (&logits + bias).map_err(|err| {
                SanitizerError::ModelLoadError(format!("Classifier bias addition failed: {err}"))
            })?;
        }

        // 5. Calculate threat probability
        let logits_vec: Vec<f32> = logits.to_vec1().map_err(|err| {
            SanitizerError::ModelLoadError(format!("Failed reading logits to vector: {err}"))
        })?;

        if logits_vec.len() >= 2 {
            // Binary classification: [prob_benign, prob_threat]
            let l0 = logits_vec[0];
            let l1 = logits_vec[1];
            let max_l = l0.max(l1);
            let e0 = (l0 - max_l).exp();
            let e1 = (l1 - max_l).exp();
            let p1 = e1 / (e0 + e1);
            Ok(p1.clamp(0.0, 1.0))
        } else if logits_vec.len() == 1 {
            // Single logit sigmoid
            let l = logits_vec[0];
            let p = 1.0 / (1.0 + (-l).exp());
            Ok(p.clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }
}

impl SemanticGuard for CandleGuard {
    fn evaluate_threat(&self, text: &str) -> Result<f32, SanitizerError> {
        self.evaluate(text)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn candle_guard_loads_micro_fixture_and_scores_threat() {
        let guard = CandleGuard::from_files(
            "tests/fixtures/ml/micro_model.safetensors",
            "tests/fixtures/ml/tokenizer.json",
        )
        .expect("micro fixture must load cleanly");

        let benign_score = guard.evaluate("hello world").expect("benign score");
        assert!(
            benign_score < 0.15,
            "Benign score {benign_score} should be < 0.15"
        );

        let threat_score = guard
            .evaluate("malicious injection attack")
            .expect("threat score");
        assert!(
            threat_score >= 0.85,
            "Threat score {threat_score} should be >= 0.85"
        );
    }
}
