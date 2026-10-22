//! Pure-Rust Shannon entropy calculation and standalone high-entropy secret detection (`FR-1`, `NFR-6`).
//!
//! Flags unanchored, standalone secret tokens with length $\ge 20$ and Shannon entropy $H \ge 4.5$
//! while avoiding false positives on normal URLs, file paths, shadow tokens, and standard code identifiers.

use crate::arena::{Span, SpanArena, SpanKind};
use crate::guard::REDACTED_CREDENTIAL_MASK;

/// Minimum byte length for standalone high-entropy secret detection (`FR-1`).
pub const MIN_ENTROPY_TOKEN_LEN: usize = 20;

/// Minimum Shannon entropy (bits per byte) for standalone secret detection (`FR-1`).
pub const MIN_SHANNON_ENTROPY: f64 = 4.5;

/// Computes the byte-level Shannon entropy $H(X) = -\sum P(x) \log_2 P(x)$ of `token`.
#[must_use]
pub fn shannon_entropy(token: &str) -> f64 {
    let bytes = token.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[usize::from(b)] += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    let len = bytes.len() as f64;
    let mut entropy = 0.0_f64;
    for &count in &counts {
        if count > 0 {
            let p = f64::from(count) / len;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Returns `true` if `token` is a candidate standalone high-entropy secret (`len >= 20`, `H >= 4.5`,
/// mixed character classes, and not a benign URL, path, shadow token, or snake/camel identifier).
#[must_use]
pub fn is_high_entropy_secret_token(token: &str) -> bool {
    if token.len() < MIN_ENTROPY_TOKEN_LEN || !token.is_ascii() {
        return false;
    }

    // Ignore already-redacted masks, shadow tokens, OpenAI call IDs (`call_...`), URLs, and file paths.
    if token.contains("REDACTED_CREDENTIAL")
        || token.starts_with("<SHADOW:")
        || token.starts_with("call_")
        || token.starts_with("http://")
        || token.starts_with("https://")
        || token.starts_with("ftp://")
        || token.starts_with("ws://")
        || token.starts_with("wss://")
        || token.starts_with('/')
        || token.contains("://")
    {
        return false;
    }

    // Ignore regular email addresses (`user@domain.tld`) where `local` only contains valid email local-part characters,
    // while still catching high-entropy passwords/tokens that include `@` alongside non-email symbols (`$`, `!`, `*`, etc.).
    if let Some((local, domain)) = token.split_once('@') {
        if !local.is_empty()
            && !domain.contains('@')
            && local.bytes().all(crate::guard::is_email_local_byte)
            && domain.contains('.')
            && domain
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
        {
            return false;
        }
    }

    // Must contain at least one digit AND both uppercase and lowercase ASCII letters
    // (prevents flagging long snake_case/camelCase identifiers or prose words).
    let mut has_upper = false;
    let mut has_lower = false;
    let mut has_digit = false;
    for b in token.bytes() {
        if b.is_ascii_uppercase() {
            has_upper = true;
        } else if b.is_ascii_lowercase() {
            has_lower = true;
        } else if b.is_ascii_digit() {
            has_digit = true;
        }
    }
    if !(has_upper && has_lower && has_digit) {
        return false;
    }

    shannon_entropy(token) >= MIN_SHANNON_ENTROPY
}

#[inline]
const fn is_token_delimiter(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t'
            | b'\n'
            | b'\r'
            | b'"'
            | b'\''
            | b'`'
            | b','
            | b';'
            | b'('
            | b')'
            | b'['
            | b']'
            | b'{'
            | b'}'
            | b'<'
            | b'>'
            | b'='
            | b':'
    )
}

/// Scans `arena.source()` in $O(N)$ linear time for standalone high-entropy secret tokens
/// (`len >= 20`, `H >= 4.5`) and pushes `SpanKind::Redacted("[REDACTED_CREDENTIAL]")`.
pub fn scan_high_entropy_tokens(arena: &mut SpanArena<'_>) {
    let src = arena.source();
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut idx = 0usize;

    while idx < len {
        while idx < len && is_token_delimiter(bytes[idx]) {
            idx += 1;
        }
        let start = idx;
        while idx < len && !is_token_delimiter(bytes[idx]) {
            idx += 1;
        }
        let end = idx;
        if end.saturating_sub(start) >= MIN_ENTROPY_TOKEN_LEN {
            if let Some(candidate) = src.get(start..end) {
                if is_high_entropy_secret_token(candidate) {
                    let _ = arena.push(Span::new(
                        start,
                        end,
                        SpanKind::Redacted(REDACTED_CREDENTIAL_MASK),
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_shannon_entropy_correctly() {
        assert!((shannon_entropy("") - 0.0).abs() < f64::EPSILON);
        assert!((shannon_entropy("aaaaaaaaaa") - 0.0).abs() < f64::EPSILON);
        assert!((shannon_entropy("abababab") - 1.0).abs() < 1e-9);
        let secret = "aB3$dE6&gH9*jK2!mN5@pQ8^rT1#uW4%";
        assert!(shannon_entropy(secret) >= MIN_SHANNON_ENTROPY);
        assert!(is_high_entropy_secret_token(secret));
        assert!(is_high_entropy_secret_token("aB3$dE6&gH9*jK2!mN5@pQ8.com"));
    }

    #[test]
    fn ignores_benign_identifiers_urls_and_short_tokens() {
        assert!(!is_high_entropy_secret_token("short123A!"));
        assert!(!is_high_entropy_secret_token(
            "alice.smith+tag123@acme-corp.com"
        ));
        assert!(!is_high_entropy_secret_token(
            "call_aBcDeFgHiJkLmNoPqRsTuVwX1234"
        ));
        assert!(!is_high_entropy_secret_token(
            "https://example.com/api/v1/users/1234567890"
        ));
        assert!(!is_high_entropy_secret_token(
            "very_long_snake_case_identifier_name_without_mixed_case_123"
        ));
        assert!(!is_high_entropy_secret_token(
            "VeryLongCamelCaseClassNameWithoutDigitsOrEntropy"
        ));

        let src =
            "const KEY_COUNT = 5; let token_limit = 4096; Bearer aB3$dE6&gH9*jK2!mN5@pQ8^rT1#uW4%";
        let mut arena = SpanArena::new(src);
        scan_high_entropy_tokens(&mut arena);
        assert_eq!(
            arena.materialize(),
            "const KEY_COUNT = 5; let token_limit = 4096; Bearer [REDACTED_CREDENTIAL]"
        );
    }
}
