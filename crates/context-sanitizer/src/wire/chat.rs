//! OpenAI-compatible wire models for `POST /v1/chat/completions` and gateway alias `POST /v1/chat` (`FR-10`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::guard::{sanitize_str_with_vault, VaultHandle};
use crate::SanitizerError;

/// OpenAI `/v1/chat/completions` and `/v1/chat` request payload (`FR-10`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Target model identifier (optional in some gateway proxies).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Ordered conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Optional OpenAI function/tool schema definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Unknown or provider-specific pass-through fields (`temperature`, `stream`, `metadata`, etc.).
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A single message in an OpenAI chat array (`FR-10`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role (`"system"`, `"user"`, `"assistant"`, `"tool"`, `"developer"`).
    pub role: String,
    /// Message content (either a string, an array of content parts, or `null` when `tool_calls` are present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatContent>,
    /// Optional assistant tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    /// Unknown or provider-specific message fields (`name`, `tool_call_id`, etc.).
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// OpenAI message `content` field supporting string or multimodal/structured part arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    /// Standard string message content.
    Text(String),
    /// Multimodal or structured array of content blocks (e.g. `[{"type": "text", "text": "..."}]`).
    Parts(Vec<Value>),
}

impl ChatCompletionRequest {
    /// Sanitizes all message contents, tool call arguments, and string values in-place using Tier 1 (`L0`).
    pub fn sanitize_in_place(&mut self, vault: &VaultHandle) {
        for msg in &mut self.messages {
            if let Some(content) = &mut msg.content {
                match content {
                    ChatContent::Text(text) => {
                        *text = sanitize_str_with_vault(text, vault);
                    }
                    ChatContent::Parts(parts) => {
                        for part in parts {
                            sanitize_json_value(part, vault);
                        }
                    }
                }
            }
            if let Some(tool_calls) = &mut msg.tool_calls {
                for tc in tool_calls {
                    sanitize_tool_call_value(tc, vault);
                }
            }
            for (k, v) in &mut msg.extra {
                if k != "tool_call_id" && k != "id" {
                    sanitize_json_value(v, vault);
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

fn sanitize_tool_call_value(tc: &mut Value, vault: &VaultHandle) {
    if let Value::Object(map) = tc {
        for (k, v) in map.iter_mut() {
            if k == "id" || k == "type" {
                continue;
            }
            if k == "function" {
                if let Value::Object(fn_map) = v {
                    for (fk, fv) in fn_map.iter_mut() {
                        if fk != "name" {
                            sanitize_json_value(fv, vault);
                        }
                    }
                }
            } else {
                sanitize_json_value(v, vault);
            }
        }
    } else {
        sanitize_json_value(tc, vault);
    }
}

pub(crate) fn sanitize_json_value(val: &mut Value, vault: &VaultHandle) {
    match val {
        Value::String(s) => {
            *s = sanitize_str_with_vault(s, vault);
        }
        Value::Array(arr) => {
            for item in arr {
                sanitize_json_value(item, vault);
            }
        }
        Value::Object(map) => {
            for v in map.values_mut() {
                sanitize_json_value(v, vault);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// Deserializes an OpenAI `/v1/chat/completions` or `/v1/chat` JSON payload, runs Tier 1 sanitization,
/// and re-serializes it with 100% schema and pass-through field parity (`FR-10`).
///
/// # Errors
/// Returns [`SanitizerError::WireParseError`] if `payload_json` is not a valid chat request JSON object.
pub fn sanitize_chat_completions_json(
    payload_json: &str,
    vault: &VaultHandle,
) -> Result<String, SanitizerError> {
    let mut req: ChatCompletionRequest = serde_json::from_str(payload_json)
        .map_err(|e| SanitizerError::WireParseError(e.to_string()))?;
    req.sanitize_in_place(vault);
    serde_json::to_string(&req).map_err(|e| SanitizerError::WireParseError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_completion_round_trip_preserves_tools_and_extra_fields() {
        let raw_json = r#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Key AKIAIOSFODNN7EXAMPLE for alice@acme.com", "name": "alice"},
                {"role": "assistant", "content": [{"type": "text", "text": "Checking 192.168.1.42"}], "tool_calls": [{"id": "call_1", "function": {"name": "db", "arguments": "{\"email\":\"alice@acme.com\"}"}}]}
            ],
            "tools": [{"type": "function", "function": {"name": "db"}}],
            "temperature": 0.2,
            "stream": false
        }"#;

        let vault = VaultHandle::with_destinations(["db"]);
        let out_json = sanitize_chat_completions_json(raw_json, &vault);
        assert!(out_json.is_ok());
        let out_str = out_json.unwrap_or_default();

        assert!(out_str.contains("[REDACTED_CREDENTIAL]"));
        assert!(!out_str.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!out_str.contains("alice@acme.com"));
        assert!(!out_str.contains("192.168.1.42"));
        assert!(out_str.contains("\"temperature\":0.2"));
        assert!(out_str.contains("\"stream\":false"));
        assert!(out_str.contains("\"tools\""));
        assert!(out_str.contains("\"name\":\"alice\""));
    }

    #[test]
    fn malformed_chat_json_returns_wire_parse_error() {
        let vault = VaultHandle::new();
        let err = sanitize_chat_completions_json("{not_valid_json", &vault);
        assert!(matches!(err, Err(SanitizerError::WireParseError(_))));
    }
}
