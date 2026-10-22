//! OpenAI-compatible wire models for `POST /v1/chat/completions` and gateway alias `POST /v1/chat` (`FR-10`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::arena::SpanArena;
use crate::compress::{compress_text_with_arena, minify_tool_definitions, CompressionStats};
use crate::guard::{run_tier1_guard, sanitize_str_with_vault, VaultHandle};
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

    /// Minifies the `tools` schema array in-place, stripping prose descriptions and redundant fields.
    pub fn minify_tools(&mut self) -> (usize, usize) {
        if let Some(tools) = &mut self.tools {
            minify_tool_definitions(tools)
        } else {
            (0, 0)
        }
    }

    /// Sanitizes and statistically compresses the chat request in-place (`Tier 1 + Tier 2`).
    ///
    /// - Auto-anchors system prompt directives (`role: "system"` or `"developer"`) and the latest user query.
    /// - Prunes older conversational turns using submodular knapsack selection.
    /// - Minifies tool schemas if present.
    pub fn sanitize_and_compress_in_place(
        &mut self,
        vault: &VaultHandle,
        target_rate: f32,
    ) -> CompressionStats {
        // 1. Minify tools if present
        let (orig_tool_bytes, min_tool_bytes) = self.minify_tools();

        // 2. Identify the tail user query for BM25 auto-anchoring
        let mut tail_query_text = String::new();
        let mut latest_user_idx = None;
        for (i, msg) in self.messages.iter().enumerate() {
            if msg.role == "user" {
                latest_user_idx = Some(i);
            }
        }

        if let Some(idx) = latest_user_idx {
            match &self.messages[idx].content {
                Some(ChatContent::Text(text)) => {
                    tail_query_text = text.clone();
                }
                Some(ChatContent::Parts(parts)) => {
                    for part in parts {
                        if part.get("type").and_then(|v| v.as_str()) == Some("text") {
                            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                if !tail_query_text.is_empty() {
                                    tail_query_text.push(' ');
                                }
                                tail_query_text.push_str(t);
                            }
                        }
                    }
                }
                None => {}
            }
        }

        let tail_query = if tail_query_text.is_empty() {
            None
        } else {
            Some(tail_query_text.as_str())
        };

        let mut total_original_bytes = orig_tool_bytes;
        let mut total_compressed_bytes = min_tool_bytes;
        let mut total_syntax_locked = 0usize;
        let mut total_pruned_spans = 0usize;
        let mut any_fail_open = false;
        let mut total_elapsed_ns = 0u64;

        // 3. Process each message
        for (idx, msg) in self.messages.iter_mut().enumerate() {
            let is_system = msg.role == "system" || msg.role == "developer";
            let is_latest_user = Some(idx) == latest_user_idx;

            if let Some(content) = &mut msg.content {
                match content {
                    ChatContent::Text(text) => {
                        let original_len = text.len();
                        total_original_bytes += original_len;

                        if is_system || is_latest_user {
                            // Auto-anchor: retain intact with infinite priority, apply Tier 1 guard only!
                            *text = sanitize_str_with_vault(text, vault);
                            total_compressed_bytes += text.len();
                        } else {
                            // Historical conversation turn: execute Tier 1 + Tier 2 pipeline
                            let mut arena = SpanArena::new(text);
                            run_tier1_guard(&mut arena, vault);
                            let (compressed, stats) =
                                compress_text_with_arena(&mut arena, target_rate, tail_query);

                            total_syntax_locked += stats.syntax_locked_count;
                            total_pruned_spans += stats.pruned_spans_count;
                            total_elapsed_ns += stats.elapsed_ns;
                            if stats.fail_open_triggered {
                                any_fail_open = true;
                            }

                            *text = compressed;
                            total_compressed_bytes += text.len();
                        }
                    }
                    ChatContent::Parts(parts) => {
                        let orig_len: usize = parts
                            .iter()
                            .map(|p| serde_json::to_vec(p).map(|v| v.len()).unwrap_or(0))
                            .sum();
                        total_original_bytes += orig_len;
                        for part in parts.iter_mut() {
                            sanitize_json_value(part, vault);
                        }
                        let comp_len: usize = parts
                            .iter()
                            .map(|p| serde_json::to_vec(p).map(|v| v.len()).unwrap_or(0))
                            .sum();
                        total_compressed_bytes += comp_len;
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

        let original_tokens = total_original_bytes.div_ceil(4);
        let compressed_tokens = total_compressed_bytes.div_ceil(4);
        let reduction_ratio = if original_tokens > 0 && compressed_tokens < original_tokens {
            ((original_tokens - compressed_tokens) as f32) / (original_tokens as f32)
        } else {
            0.0
        };

        CompressionStats {
            original_bytes: total_original_bytes,
            compressed_bytes: total_compressed_bytes,
            original_tokens,
            compressed_tokens,
            reduction_ratio,
            syntax_locked_count: total_syntax_locked,
            pruned_spans_count: total_pruned_spans,
            fail_open_triggered: any_fail_open,
            elapsed_ns: total_elapsed_ns,
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

/// Deserializes an OpenAI chat payload, executes both Tier 1 deterministic guard/vault passes
/// and Tier 2 syntax-safe statistical context and schema minification passes, returning `(json, stats)`.
///
/// # Errors
/// Returns [`SanitizerError::WireParseError`] if `payload_json` is not a valid chat request JSON object.
pub fn sanitize_and_compress_chat_completions_json(
    payload_json: &str,
    vault: &VaultHandle,
    target_rate: f32,
) -> Result<(String, CompressionStats), SanitizerError> {
    let mut req: ChatCompletionRequest = serde_json::from_str(payload_json)
        .map_err(|e| SanitizerError::WireParseError(e.to_string()))?;
    let stats = req.sanitize_and_compress_in_place(vault, target_rate);
    let serialized =
        serde_json::to_string(&req).map_err(|e| SanitizerError::WireParseError(e.to_string()))?;
    Ok((serialized, stats))
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

    #[test]
    fn sanitize_and_compress_chat_json_minifies_tools_and_compresses_history() {
        let raw_json = r#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Here is some old historical text: it is truly very clear that the user requested details regarding the system setup."},
                {"role": "assistant", "content": "The system setup was completed previously."},
                {"role": "user", "content": "Please query the database for the active credentials."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Verbose tool description for query_database that should be minified away.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The SQL query to run against customer DB."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }]
        }"#;

        let vault = VaultHandle::new();
        let res = sanitize_and_compress_chat_completions_json(raw_json, &vault, 0.60);
        assert!(res.is_ok());
        let (compressed_json, stats) = res.unwrap_or_default();

        // System prompt and latest query are preserved intact
        assert!(compressed_json.contains("You are a helpful assistant."));
        assert!(compressed_json.contains("Please query the database for the active credentials."));

        // Tool description is minified away
        assert!(!compressed_json.contains("Verbose tool description"));
        assert!(compressed_json.contains("query_database"));

        // Reduction achieved
        assert!(stats.reduction_ratio > 0.0);
    }
}
