#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(unexpected_cfgs)]

//! Universal WebAssembly and WASI bindings for `context-sanitizer` (`Story 5.1`, `FR13`, `AD-3`).
//!
//! Exposes client-side Tier 1 credential/PII sanitization, CSPRNG-isolated session shadow vaulting,
//! Stage 3A fast-breaking, and Tier 2 statistical compression with zero native C/C++ dependencies.

use wasm_bindgen::prelude::*;

use context_sanitizer::arena::SpanArena;
use context_sanitizer::compress::compress_text_with_arena;
use context_sanitizer::guard::{run_tier1_guard, VaultHandle};
use context_sanitizer::ml::scan_fast_breaker;
use context_sanitizer::wire::{
    ChatCompletionRequest, ChatContent, ChatMessage, CompletionPrompt, CompletionRequest,
};
use context_sanitizer::SanitizerError;

/// Maps a core [`SanitizerError`] to a typed JavaScript Error (`AD-2`, `FR13`).
fn map_core_err(err: SanitizerError) -> JsValue {
    match err {
        SanitizerError::ThreatBlocked { score_bps, reason } => {
            let msg = format!("threat detected (score: {score_bps} bps): {reason}");
            let js_err = js_sys::Error::new(&msg);
            js_err.set_name("InjectionBlockedError");
            let _ = js_sys::Reflect::set(
                &js_err,
                &JsValue::from_str("reason"),
                &JsValue::from_str(&reason),
            );
            let _ = js_sys::Reflect::set(
                &js_err,
                &JsValue::from_str("score"),
                &JsValue::from_f64(f64::from(score_bps) / 10_000.0),
            );
            let _ = js_sys::Reflect::set(
                &js_err,
                &JsValue::from_str("score_bps"),
                &JsValue::from_f64(f64::from(score_bps)),
            );
            js_err.into()
        }
        SanitizerError::UnauthorizedRehydration(dest) => {
            let msg = format!("unauthorized rehydration destination: {dest}");
            let js_err = js_sys::Error::new(&msg);
            js_err.set_name("UnauthorizedRehydrationError");
            let _ = js_sys::Reflect::set(
                &js_err,
                &JsValue::from_str("destination"),
                &JsValue::from_str(&dest),
            );
            js_err.into()
        }
        other => js_sys::Error::new(&other.to_string()).into(),
    }
}

/// Helper function to scan fast breaker across a slice of chat messages.
fn check_messages_fast_breaker(messages: &[ChatMessage]) -> Result<(), SanitizerError> {
    for msg in messages {
        if let Some(content) = &msg.content {
            match content {
                ChatContent::Text(t) => {
                    scan_fast_breaker(t)?;
                }
                ChatContent::Parts(parts) => {
                    for part in parts {
                        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                            scan_fast_breaker(t)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Helper function to scan fast breaker across completion prompt text.
fn check_completion_fast_breaker(req: &CompletionRequest) -> Result<(), SanitizerError> {
    match &req.prompt {
        CompletionPrompt::Single(s) => scan_fast_breaker(s)?,
        CompletionPrompt::Batch(batch) => {
            for s in batch {
                scan_fast_breaker(s)?;
            }
        }
    }
    Ok(())
}

/// Opaque memory-safe handle wrapping the cryptographic session vault (`AD-3`, `FR13`, `NFR2`).
#[wasm_bindgen]
pub struct WasmVaultHandle {
    inner: Option<VaultHandle>,
}

#[wasm_bindgen]
impl WasmVaultHandle {
    /// Internal constructor for creating a new active vault handle.
    #[wasm_bindgen(constructor)]
    pub fn new_empty() -> Self {
        Self { inner: None }
    }

    /// Rehydrates `<SHADOW:...>` tokens in tool arguments back to original PII values
    /// if and only if `destination` matches the session whitelist (`FR3`, `FR13`).
    pub fn rehydrate(&self, tool_args: &JsValue, destination: &str) -> Result<JsValue, JsValue> {
        let vault = self.inner.as_ref().ok_or_else(|| {
            let err = js_sys::Error::new("VaultHandle has been freed and cannot be accessed");
            err.set_name("Error");
            JsValue::from(err)
        })?;

        if let Some(tool_str) = tool_args.as_string() {
            let rehydrated = vault
                .rehydrate(&tool_str, destination)
                .map_err(map_core_err)?;
            Ok(JsValue::from_str(&rehydrated))
        } else if tool_args.is_object() {
            let json_str = js_sys::JSON::stringify(tool_args).map_err(|e| {
                js_sys::Error::new(&format!("failed to stringify tool_args: {e:?}"))
            })?;
            let s = json_str.as_string().unwrap_or_default();
            let rehydrated = vault.rehydrate(&s, destination).map_err(map_core_err)?;
            let parsed = js_sys::JSON::parse(&rehydrated).map_err(|e| {
                js_sys::Error::new(&format!("failed to parse rehydrated JSON: {e:?}"))
            })?;
            Ok(parsed)
        } else {
            Err(js_sys::TypeError::new("tool_args must be a JSON string or object").into())
        }
    }

    /// Explicitly destroys the vault, immediately dropping all session salt and tokens (`NFR2`).
    pub fn destroy(mut self) {
        self.inner.take();
    }
}

impl Drop for WasmVaultHandle {
    fn drop(&mut self) {
        self.inner.take();
    }
}

/// Minifies OpenAI JSON tool definitions in-place, stripping redundant descriptions (`FR-6`).
#[wasm_bindgen]
pub fn minify_tools(tools_json: &str) -> Result<String, JsValue> {
    let mut val: serde_json::Value = serde_json::from_str(tools_json)
        .map_err(|e| js_sys::Error::new(&format!("invalid JSON in minify_tools: {e}")))?;
    context_sanitizer::compress::schema::minify_value(&mut val);
    serde_json::to_string(&val)
        .map_err(|e| js_sys::Error::new(&format!("serialization error: {e}")).into())
}

/// Sanitizes a raw text string, chat completions JSON object/string, or completion payload (`FR13`, `AD-1`).
///
/// Returns an object `{ sanitizedJson: string, vault: WasmVaultHandle }`.
#[wasm_bindgen]
pub fn sanitize(
    input: &JsValue,
    rate: Option<f32>,
    whitelist: Option<Vec<String>>,
) -> Result<JsValue, JsValue> {
    let target_rate = rate.unwrap_or(1.0);
    if !(0.0..=1.0).contains(&target_rate) || target_rate.is_nan() {
        return Err(
            js_sys::RangeError::new("rate must be a valid float between 0.0 and 1.0").into(),
        );
    }

    let destinations = whitelist.unwrap_or_default();
    let vault = VaultHandle::with_destinations(destinations);

    let input_str = if let Some(s) = input.as_string() {
        s
    } else if input.is_object() {
        let s = js_sys::JSON::stringify(input)
            .map_err(|e| js_sys::Error::new(&format!("failed to stringify object: {e:?}")))?;
        s.as_string().unwrap_or_default()
    } else {
        return Err(js_sys::TypeError::new("input must be a string or JSON object").into());
    };

    let trimmed = input_str.trim_start();
    let is_json = trimmed.starts_with('{');

    if is_json {
        if let Ok(parsed_val) = serde_json::from_str::<serde_json::Value>(&input_str) {
            if let Ok(mut req) = serde_json::from_value::<ChatCompletionRequest>(parsed_val.clone())
            {
                check_messages_fast_breaker(&req.messages).map_err(map_core_err)?;
                if target_rate < 1.0 {
                    let _stats = req.sanitize_and_compress_in_place(&vault, target_rate);
                } else {
                    req.sanitize_in_place(&vault);
                }
                let clean_json =
                    serde_json::to_string(&req).map_err(|e| js_sys::Error::new(&e.to_string()))?;

                let result = js_sys::Object::new();
                js_sys::Reflect::set(
                    &result,
                    &JsValue::from_str("sanitizedJson"),
                    &JsValue::from_str(&clean_json),
                )?;
                js_sys::Reflect::set(
                    &result,
                    &JsValue::from_str("vault"),
                    &JsValue::from(WasmVaultHandle { inner: Some(vault) }),
                )?;
                return Ok(result.into());
            } else if let Ok(mut comp) = serde_json::from_value::<CompletionRequest>(parsed_val) {
                check_completion_fast_breaker(&comp).map_err(map_core_err)?;
                comp.sanitize_in_place(&vault);
                let clean_json =
                    serde_json::to_string(&comp).map_err(|e| js_sys::Error::new(&e.to_string()))?;

                let result = js_sys::Object::new();
                js_sys::Reflect::set(
                    &result,
                    &JsValue::from_str("sanitizedJson"),
                    &JsValue::from_str(&clean_json),
                )?;
                js_sys::Reflect::set(
                    &result,
                    &JsValue::from_str("vault"),
                    &JsValue::from(WasmVaultHandle { inner: Some(vault) }),
                )?;
                return Ok(result.into());
            }
        }
    }

    // Fall back to raw text sanitization path
    scan_fast_breaker(&input_str).map_err(map_core_err)?;
    let mut arena = SpanArena::new(&input_str);
    run_tier1_guard(&mut arena, &vault);
    let candidate = if target_rate < 1.0 {
        let (compressed, _) = compress_text_with_arena(&mut arena, target_rate, None);
        compressed
    } else {
        arena.materialize()
    };
    scan_fast_breaker(&candidate).map_err(map_core_err)?;

    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("sanitizedJson"),
        &JsValue::from_str(&candidate),
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("vault"),
        &JsValue::from(WasmVaultHandle { inner: Some(vault) }),
    )?;
    Ok(result.into())
}
