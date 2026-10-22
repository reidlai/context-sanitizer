//! OpenAI-compatible wire models for legacy `POST /v1/completions` payloads (`FR-10`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::guard::{sanitize_str_with_vault, VaultHandle};
use crate::wire::chat::sanitize_json_value;
use crate::SanitizerError;

/// OpenAI `/v1/completions` request payload (`FR-10`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Target model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Prompt text or array of prompt strings.
    pub prompt: CompletionPrompt,
    /// Unknown or provider-specific pass-through fields (`max_tokens`, `temperature`, `suffix`, etc.).
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Prompt value accepted by `/v1/completions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    /// Single string prompt.
    Single(String),
    /// Batch array of string prompts.
    Batch(Vec<String>),
}

impl CompletionRequest {
    /// Sanitizes the prompt string(s) and extra string fields (such as `suffix`) in-place using Tier 1 (`L0`).
    pub fn sanitize_in_place(&mut self, vault: &VaultHandle) {
        match &mut self.prompt {
            CompletionPrompt::Single(text) => {
                *text = sanitize_str_with_vault(text, vault);
            }
            CompletionPrompt::Batch(items) => {
                for item in items {
                    *item = sanitize_str_with_vault(item, vault);
                }
            }
        }
        for (k, v) in &mut self.extra {
            if k != "id" && k != "user" {
                sanitize_json_value(v, vault);
            }
        }
    }
}

/// Deserializes an OpenAI `/v1/completions` JSON payload, runs Tier 1 sanitization,
/// and re-serializes it with 100% schema and pass-through field parity (`FR-10`).
///
/// # Errors
/// Returns [`SanitizerError::WireParseError`] if `payload_json` is not a valid completion request JSON object.
pub fn sanitize_completions_json(
    payload_json: &str,
    vault: &VaultHandle,
) -> Result<String, SanitizerError> {
    let mut req: CompletionRequest = serde_json::from_str(payload_json)
        .map_err(|e| SanitizerError::WireParseError(e.to_string()))?;
    req.sanitize_in_place(vault);
    serde_json::to_string(&req).map_err(|e| SanitizerError::WireParseError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_single_and_batch_prompts_sanitize_cleanly() {
        let single_json = r#"{
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Deploy with ghp_1234567890abcdefghijklmnopqrstuv for bob@acme.com",
            "max_tokens": 128
        }"#;
        let vault = VaultHandle::with_destinations(["deploy_tool"]);
        let out = sanitize_completions_json(single_json, &vault).unwrap_or_default();
        assert!(out.contains("[REDACTED_CREDENTIAL]"));
        assert!(!out.contains("ghp_1234567890abcdefghijklmnopqrstuv"));
        assert!(out.contains("<SHADOW:EMAIL_"));
        assert!(out.contains("\"max_tokens\":128"));

        let batch_json = r#"{
            "prompt": ["Call +1-415-555-0199", "Secret api_key = \"super_secret_key_value_123\""],
            "suffix": "Contact bob@acme.com with AKIAIOSFODNN7EXAMPLE"
        }"#;
        let out_batch = sanitize_completions_json(batch_json, &vault).unwrap_or_default();
        assert!(out_batch.contains("<SHADOW:PHONE_"));
        assert!(out_batch.contains("<SHADOW:EMAIL_"));
        assert!(out_batch.contains("[REDACTED_CREDENTIAL]"));
        assert!(!out_batch.contains("AKIAIOSFODNN7EXAMPLE"));
    }

    #[test]
    fn malformed_completion_json_returns_wire_parse_error() {
        let vault = VaultHandle::new();
        let err = sanitize_completions_json("{not_valid_json", &vault);
        assert!(matches!(err, Err(SanitizerError::WireParseError(_))));
    }
}
