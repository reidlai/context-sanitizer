#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! PyO3 Zero-C Python bindings for `context-sanitizer` (`Story 4.1`, `FR11`, `AD-2`).
//!
//! Exposes nanosecond FFI with GIL-free execution (`py.allow_threads`), opaque memory-safe
//! [`VaultHandle`] serialization guards, and typed exception hierarchies.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use context_sanitizer::arena::SpanArena;
use context_sanitizer::compress::{compress_text_with_arena, minify_tool_definitions};
use context_sanitizer::guard::{run_tier1_guard, PiiKind, VaultHandle as CoreVaultHandle};
use context_sanitizer::ml::scan_fast_breaker;
use context_sanitizer::wire::{ChatCompletionRequest, ChatContent};
use context_sanitizer::SanitizerError as CoreSanitizerError;

create_exception!(_core, SanitizerError, PyException);
create_exception!(_core, UnauthorizedRehydrationError, SanitizerError);
create_exception!(_core, InjectionBlockedError, SanitizerError);

fn map_core_err(py: Python<'_>, err: CoreSanitizerError) -> PyErr {
    match err {
        CoreSanitizerError::ThreatBlocked { score_bps, reason } => {
            let msg = format!("threat detected (score: {score_bps} bps): {reason}");
            let py_err = InjectionBlockedError::new_err(msg);
            let val = py_err.value(py);
            let _ = val.setattr("score_bps", score_bps);
            let _ = val.setattr("score", f64::from(score_bps) / 10_000.0);
            let _ = val.setattr("reason", reason);
            py_err
        }
        CoreSanitizerError::UnauthorizedRehydration(dest) => {
            let msg = format!("unauthorized rehydration destination: {dest}");
            let py_err = UnauthorizedRehydrationError::new_err(msg);
            let val = py_err.value(py);
            let _ = val.setattr("destination", dest);
            py_err
        }
        other => SanitizerError::new_err(other.to_string()),
    }
}

/// Opaque memory-safe handle wrapping the cryptographic session vault (`AD-3`, `FR11`, `NFR5`).
#[pyclass]
pub struct VaultHandle {
    inner: Arc<CoreVaultHandle>,
}

#[pymethods]
impl VaultHandle {
    /// Safe diagnostic string representation preventing credential/salt leakage (`FR11`, `NFR5`).
    fn __repr__(&self) -> &'static str {
        "<VaultHandle session_active=True>"
    }

    /// Safe string representation.
    fn __str__(&self) -> &'static str {
        "<VaultHandle session_active=True>"
    }

    /// Rejects pickle serialization with explicit `TypeError` (`FR11`, `NFR5`).
    fn __reduce__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err("VaultHandle cannot be pickled"))
    }

    /// Explicitly denies `__dict__` inspection attribute with `TypeError` (`FR11`, `NFR5`).
    #[getter]
    fn __dict__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err("VaultHandle does not expose __dict__"))
    }

    /// Rehydrates `<SHADOW:...>` tokens in `tool_args` (dict or JSON string) back to original
    /// PII values if and only if `destination` matches the session whitelist (`FR3`, `FR11`).
    #[pyo3(signature = (tool_args, destination))]
    fn rehydrate(
        &self,
        py: Python<'_>,
        tool_args: &Bound<'_, PyAny>,
        destination: &str,
    ) -> PyResult<PyObject> {
        let vault = Arc::clone(&self.inner);
        let dest = destination.to_string();

        if let Ok(py_str) = tool_args.extract::<String>() {
            let res = py.allow_threads(move || vault.rehydrate(&py_str, &dest));
            match res {
                Ok(rehydrated) => Ok(rehydrated.into_pyobject(py)?.into_any().unbind()),
                Err(err) => Err(map_core_err(py, err)),
            }
        } else if tool_args.is_instance_of::<PyDict>() || tool_args.is_instance_of::<PyList>() {
            let json_mod = py.import("json")?;
            let json_str: String = json_mod.call_method1("dumps", (tool_args,))?.extract()?;
            let res = py.allow_threads(move || vault.rehydrate(&json_str, &dest));
            match res {
                Ok(rehydrated) => {
                    let parsed = json_mod.call_method1("loads", (rehydrated,))?;
                    Ok(parsed.unbind())
                }
                Err(err) => Err(map_core_err(py, err)),
            }
        } else {
            Err(PyTypeError::new_err(
                "tool_args must be a string, dict, or list",
            ))
        }
    }

    /// Registers an additional authorized tool destination.
    fn allow_destination(&self, destination: &str) {
        self.inner.allow_destination(destination);
    }

    /// Returns the number of unique PII entities vaulted in this session.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Python `len(vault)` protocol.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Python truthiness protocol.
    fn __bool__(&self) -> bool {
        !self.inner.is_empty()
    }

    /// Returns `true` if no PII entities have been vaulted in this session.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Manually vaults an operational PII entity (`email`, `ip`, `phone`).
    #[pyo3(signature = (kind, raw_pii))]
    fn vault_pii(&self, kind: &str, raw_pii: &str) -> PyResult<String> {
        let pii_kind = match kind.to_ascii_lowercase().as_str() {
            "email" => PiiKind::Email,
            "ip" => PiiKind::Ip,
            "phone" => PiiKind::Phone,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported PII kind: {kind}; expected 'email', 'ip', or 'phone'"
                )))
            }
        };
        Ok(self.inner.vault_pii(pii_kind, raw_pii))
    }
}

/// Helper function to scan fast breaker across a slice of messages.
fn check_messages_fast_breaker(
    messages: &[context_sanitizer::wire::ChatMessage],
) -> Result<(), CoreSanitizerError> {
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

/// Sanitizes a raw string, chat completions JSON object, or list of messages, releasing the
/// Python GIL during execution (`FR11`, `AD-2`).
#[pyfunction]
#[pyo3(signature = (input, rate=1.0, whitelist=None))]
pub fn sanitize(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    rate: Option<f32>,
    whitelist: Option<Vec<String>>,
) -> PyResult<(PyObject, VaultHandle)> {
    let destinations = whitelist.unwrap_or_default();
    let vault = CoreVaultHandle::with_destinations(destinations);
    let target_rate = rate.unwrap_or(1.0);
    if !(0.0..=1.0).contains(&target_rate) || target_rate.is_nan() {
        return Err(PyValueError::new_err(
            "rate must be a valid float between 0.0 and 1.0",
        ));
    }

    // 1. Raw string input
    if let Ok(py_str) = input.extract::<String>() {
        let vault_clone = vault.clone();
        let is_json_obj = py_str.trim_start().starts_with('{');

        if is_json_obj {
            if let Ok(mut req) = serde_json::from_str::<ChatCompletionRequest>(&py_str) {
                let res = py.allow_threads(move || -> Result<String, CoreSanitizerError> {
                    check_messages_fast_breaker(&req.messages)?;
                    let _stats = req.sanitize_and_compress_in_place(&vault_clone, target_rate);
                    serde_json::to_string(&req)
                        .map_err(|e| CoreSanitizerError::WireParseError(e.to_string()))
                });
                match res {
                    Ok(clean_json) => {
                        return Ok((
                            clean_json.into_pyobject(py)?.into_any().unbind(),
                            VaultHandle {
                                inner: Arc::new(vault),
                            },
                        ));
                    }
                    Err(err) => return Err(map_core_err(py, err)),
                }
            }
        }

        // Plain text string
        let res = py.allow_threads(move || -> Result<String, CoreSanitizerError> {
            scan_fast_breaker(&py_str)?;
            let mut arena = SpanArena::new(&py_str);
            run_tier1_guard(&mut arena, &vault_clone);
            let candidate = if target_rate < 1.0 {
                let (compressed, _) = compress_text_with_arena(&mut arena, target_rate, None);
                compressed
            } else {
                arena.materialize()
            };
            scan_fast_breaker(&candidate)?;
            Ok(candidate)
        });

        match res {
            Ok(sanitized) => Ok((
                sanitized.into_pyobject(py)?.into_any().unbind(),
                VaultHandle {
                    inner: Arc::new(vault),
                },
            )),
            Err(err) => Err(map_core_err(py, err)),
        }
    } else if input.is_instance_of::<PyDict>() {
        // 2. Python dictionary representing a ChatCompletionRequest or payload
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (input,))?.extract()?;
        let vault_clone = vault.clone();

        let res = py.allow_threads(move || -> Result<String, CoreSanitizerError> {
            let mut req: ChatCompletionRequest = serde_json::from_str(&json_str)
                .map_err(|e| CoreSanitizerError::WireParseError(e.to_string()))?;
            check_messages_fast_breaker(&req.messages)?;
            let _stats = req.sanitize_and_compress_in_place(&vault_clone, target_rate);
            serde_json::to_string(&req)
                .map_err(|e| CoreSanitizerError::WireParseError(e.to_string()))
        });

        match res {
            Ok(clean_json) => {
                let parsed = json_mod.call_method1("loads", (clean_json,))?;
                Ok((
                    parsed.unbind(),
                    VaultHandle {
                        inner: Arc::new(vault),
                    },
                ))
            }
            Err(err) => Err(map_core_err(py, err)),
        }
    } else if input.is_instance_of::<PyList>() {
        // 3. Python list of messages
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (input,))?.extract()?;
        let vault_clone = vault.clone();

        let res = py.allow_threads(move || -> Result<String, CoreSanitizerError> {
            let messages: Vec<context_sanitizer::wire::ChatMessage> =
                serde_json::from_str(&json_str)
                    .map_err(|e| CoreSanitizerError::WireParseError(e.to_string()))?;
            check_messages_fast_breaker(&messages)?;

            let mut req = ChatCompletionRequest {
                model: None,
                messages,
                tools: None,
                extra: BTreeMap::new(),
            };
            let _stats = req.sanitize_and_compress_in_place(&vault_clone, target_rate);
            serde_json::to_string(&req.messages)
                .map_err(|e| CoreSanitizerError::WireParseError(e.to_string()))
        });

        match res {
            Ok(clean_json) => {
                let parsed = json_mod.call_method1("loads", (clean_json,))?;
                Ok((
                    parsed.unbind(),
                    VaultHandle {
                        inner: Arc::new(vault),
                    },
                ))
            }
            Err(err) => Err(map_core_err(py, err)),
        }
    } else {
        Err(PyTypeError::new_err(
            "input must be a string, list of messages, or dict",
        ))
    }
}

/// Minifies OpenAI function/tool definitions in-place (`FR-5`).
#[pyfunction]
pub fn minify_tools(py: Python<'_>, tools: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (tools,))?.extract()?;
    let mut parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str)
        .map_err(|e| PyValueError::new_err(format!("invalid tools JSON array: {e}")))?;

    minify_tool_definitions(&mut parsed);

    let serialized = serde_json::to_string(&parsed)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {e}")))?;
    let result = json_mod.call_method1("loads", (serialized,))?;
    Ok(result.unbind())
}

/// Python extension module `_core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<VaultHandle>()?;
    m.add("SanitizerError", py.get_type::<SanitizerError>())?;
    m.add(
        "UnauthorizedRehydrationError",
        py.get_type::<UnauthorizedRehydrationError>(),
    )?;
    m.add(
        "InjectionBlockedError",
        py.get_type::<InjectionBlockedError>(),
    )?;
    m.add_function(wrap_pyfunction!(sanitize, m)?)?;
    m.add_function(wrap_pyfunction!(minify_tools, m)?)?;
    Ok(())
}
