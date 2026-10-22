//! Wasm unit and integration tests (`Story 5.1`, `FR13`, `AD-3`).

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn test_wasm_context_sanitizer_flow() {
    use wasm_bindgen::JsValue;
    let payload = r#"{
        "messages": [
            {"role": "user", "content": "Contact alice@acme.com with key AKIAIOSFODNN7EXAMPLE"}
        ]
    }"#;
    let res = wasm::sanitize(
        &JsValue::from_str(payload),
        Some(0.8),
        Some(vec!["crm".to_string()]),
    )
    .expect("sanitization should succeed");
    let json_val = js_sys::Reflect::get(&res, &JsValue::from_str("sanitizedJson"))
        .expect("sanitizedJson field");
    let json_str = json_val.as_string().expect("should be string");
    assert!(json_str.contains("[REDACTED_CREDENTIAL]"));
    assert!(json_str.contains("<SHADOW:EMAIL_"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_native_compilation_and_exports() {
    // Verifies that wasm crate compiles and exports symbols cleanly on native targets.
    let handle = wasm::WasmVaultHandle::new_empty();
    drop(handle);
}
