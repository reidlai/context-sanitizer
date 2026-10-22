//! Linear-time $O(N)$ SIMD DFA credential and operational PII scanners (`FR-1`, `FR-2`, `NFR-1`, `AD-1`).
//!
//! Uses `aho-corasick` DFA automatons and single-pass byte state machines with zero backtracking
//! to guarantee `< 500,000 ns` P99 latency per 4KB payload even on adversarial ReDoS inputs.

use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

use crate::arena::{Span, SpanArena, SpanKind};
use crate::guard::vault::{PiiKind, VaultHandle};
use crate::guard::REDACTED_CREDENTIAL_MASK;

/// Minimum assigned value length for generalized `*_KEY`, `*_SECRET`, `*_TOKEN` heuristics (`FR-1`).
pub const MIN_ASSIGNED_SECRET_VALUE_LEN: usize = 16;

const VENDOR_PREFIXES: &[&str] = &[
    "AKIA",
    "ASIA",
    "AIza",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xoxr-",
    "sk-proj-",
    "sk-live-",
    "sk_live_",
    "rk_live_",
    "sq0csp-",
    "00D",
    "00u",
    "00o",
    "eyJhbGciOi",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN PGP PRIVATE KEY BLOCK-----",
    "-----BEGIN CERTIFICATE-----",
    "AccountKey=",
    "SharedAccessKey=",
];

const KEY_SUFFIXES: &[&str] = &[
    "_PRIVATE_KEY",
    "_CREDENTIAL",
    "_PASSWORD",
    "_APIKEY",
    "_SECRET",
    "_TOKEN",
    "_AUTH",
    "_PASS",
    "_KEY",
    "APIKEY",
    "PASSWORD",
    "CLIENT_SECRET",
    "TENANT_SECRET",
    "BEARER_TOKEN",
];

fn vendor_automaton() -> &'static AhoCorasick {
    static AC: OnceLock<AhoCorasick> = OnceLock::new();
    AC.get_or_init(|| {
        AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .ascii_case_insensitive(false)
            .build(VENDOR_PREFIXES)
            .unwrap_or_else(|_| unreachable!())
    })
}

fn suffix_automaton() -> &'static AhoCorasick {
    static AC: OnceLock<AhoCorasick> = OnceLock::new();
    AC.get_or_init(|| {
        AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .ascii_case_insensitive(true)
            .build(KEY_SUFFIXES)
            .unwrap_or_else(|_| unreachable!())
    })
}

#[inline]
const fn is_secret_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b'+' | b'=')
}

#[inline]
const fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')
}

/// Scans `arena.source()` in $O(N)$ linear time for known vendor credentials and `*_KEY` / `*_SECRET`
/// assignments ($\ge 16$ chars), pushing `SpanKind::Redacted("[REDACTED_CREDENTIAL]")`.
pub fn scan_credentials(arena: &mut SpanArena<'_>) {
    let src = arena.source();
    scan_vendor_signatures(src, arena);
    scan_key_value_suffix_assignments(src, arena);
}

fn scan_vendor_signatures(src: &str, arena: &mut SpanArena<'_>) {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let ac = vendor_automaton();
    let mut skip_until = 0usize;

    for mat in ac.find_iter(src) {
        let start = mat.start();
        if start < skip_until {
            continue;
        }
        let end = mat.end();
        let matched = &src[start..end];

        if matched.starts_with("-----BEGIN ") {
            // Scan forward linearly to the matching `-----END ...-----` block or end of string.
            let rest = &src[end..];
            let block_end = if let Some(end_idx) = rest.find("-----END ") {
                let after_end = end + end_idx + "-----END ".len();
                if let Some(close_dashes) = src.get(after_end..).and_then(|s| s.find("-----")) {
                    after_end + close_dashes + 5
                } else {
                    len
                }
            } else {
                len
            };
            let _ = arena.push(Span::new(
                start,
                block_end,
                SpanKind::Redacted(REDACTED_CREDENTIAL_MASK),
            ));
            skip_until = block_end;
            continue;
        }

        if matched == "AccountKey=" || matched == "SharedAccessKey=" {
            let val_start = end;
            let mut val_end = val_start;
            while val_end < len && is_secret_token_char(bytes[val_end]) && bytes[val_end] != b';' {
                val_end += 1;
            }
            if val_end.saturating_sub(val_start) >= MIN_ASSIGNED_SECRET_VALUE_LEN {
                let _ = arena.push(Span::new(
                    val_start,
                    val_end,
                    SpanKind::Redacted(REDACTED_CREDENTIAL_MASK),
                ));
                skip_until = val_end;
            }
            continue;
        }

        // Ensure prefix starts at a word/token boundary.
        if start > 0 && is_ident_char(bytes[start - 1]) {
            continue;
        }

        let mut tok_end = end;
        while tok_end < len && is_secret_token_char(bytes[tok_end]) {
            tok_end += 1;
        }

        let total_len = tok_end.saturating_sub(start);
        let min_len = if matched == "00D" || matched == "00u" || matched == "00o" {
            15
        } else {
            16
        };

        if total_len >= min_len && has_credential_entropy_or_vendor_structure(&src[start..tok_end])
        {
            let _ = arena.push(Span::new(
                start,
                tok_end,
                SpanKind::Redacted(REDACTED_CREDENTIAL_MASK),
            ));
            skip_until = tok_end;
        }
    }
}

#[inline]
fn has_credential_entropy_or_vendor_structure(token: &str) -> bool {
    if token.starts_with("AKIA") || token.starts_with("ASIA") {
        let tail = &token[4..];
        let all_zeros = tail.bytes().all(|b| b == b'0');
        return token.len() >= 16 && !all_zeros;
    }
    if token.starts_with("00D") || token.starts_with("00u") || token.starts_with("00o") {
        // Okta/Salesforce IDs & tokens are 15-42 alphanumeric chars and not pure lowercase hex commit hashes
        let is_pure_lowercase_hex = token
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        return (15..=42).contains(&token.len()) && !is_pure_lowercase_hex;
    }
    true
}

fn scan_key_value_suffix_assignments(src: &str, arena: &mut SpanArena<'_>) {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let ac = suffix_automaton();

    for mat in ac.find_iter(src) {
        let suffix_end = mat.end();
        if suffix_end < len && is_ident_char(bytes[suffix_end]) {
            continue;
        }

        let mut cursor = suffix_end;
        if cursor < len && (bytes[cursor] == b'"' || bytes[cursor] == b'\'') {
            cursor += 1;
        }
        while cursor < len && (bytes[cursor] == b' ' || bytes[cursor] == b'\t') {
            cursor += 1;
        }
        if cursor >= len || (bytes[cursor] != b'=' && bytes[cursor] != b':') {
            continue;
        }
        cursor += 1;
        while cursor < len && (bytes[cursor] == b' ' || bytes[cursor] == b'\t') {
            cursor += 1;
        }
        if cursor >= len {
            continue;
        }

        let (val_start, val_end) = if bytes[cursor] == b'"' || bytes[cursor] == b'\'' {
            let quote = bytes[cursor];
            let start = cursor + 1;
            let mut end = start;
            while end < len && bytes[end] != quote {
                if bytes[end] == b'\\' && end + 1 < len {
                    end += 2;
                } else {
                    end += 1;
                }
            }
            (start, end)
        } else {
            let start = cursor;
            let mut end = start;
            while end < len
                && !matches!(
                    bytes[end],
                    b' ' | b'\t' | b'\n' | b'\r' | b',' | b';' | b')' | b'}' | b']'
                )
            {
                end += 1;
            }
            (start, end)
        };

        let val_len = val_end.saturating_sub(val_start);
        if val_len >= MIN_ASSIGNED_SECRET_VALUE_LEN {
            if let Some(val_slice) = src.get(val_start..val_end) {
                if !val_slice.starts_with("<SHADOW:") && !val_slice.contains("REDACTED_CREDENTIAL")
                {
                    let _ = arena.push(Span::new(
                        val_start,
                        val_end,
                        SpanKind::Redacted(REDACTED_CREDENTIAL_MASK),
                    ));
                }
            }
        }
    }
}

/// Scans `arena.source()` in $O(N)$ linear time for operational PII (emails, IPv4/IPv6 addresses,
/// and telephone numbers), skipping any byte range already marked `SpanKind::Redacted` so root
/// credentials can never enter `VaultHandle` (`FR-1`, `FR-2`, `AD-1`, `AD-3`).
pub fn scan_and_vault_pii(arena: &mut SpanArena<'_>, vault: &VaultHandle) {
    let src = arena.source();
    scan_emails(src, arena, vault);
    scan_ips(src, arena, vault);
    scan_phones(src, arena, vault);
}

#[inline]
const fn is_email_local_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'%' | b'+' | b'-')
}

#[inline]
const fn is_email_domain_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-')
}

fn scan_emails(src: &str, arena: &mut SpanArena<'_>, vault: &VaultHandle) {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    // Track lower bound of local-part scan so backward scanning never revisits earlier bytes (`O(N)` guarantee).
    let mut min_local_bound = 0usize;

    while i < len {
        if bytes[i] == b'@' {
            let mut start = i;
            while start > min_local_bound && is_email_local_char(bytes[start - 1]) {
                start -= 1;
            }
            while start < i && bytes[start] == b'.' {
                start += 1;
            }

            let mut end = i + 1;
            while end < len && is_email_domain_char(bytes[end]) {
                end += 1;
            }
            while end > i + 1 && (bytes[end - 1] == b'.' || bytes[end - 1] == b'-') {
                end -= 1;
            }

            let local_len = i.saturating_sub(start);
            let domain = src.get((i + 1)..end).unwrap_or("");
            if local_len >= 1
                && domain.len() >= 4
                && domain.contains('.')
                && domain.rsplit('.').next().is_some_and(|tld| {
                    tld.len() >= 2 && tld.bytes().all(|b| b.is_ascii_alphabetic())
                })
            {
                if !arena.overlaps_redacted(start, end) {
                    if let Some(email) = src.get(start..end) {
                        let shadow = vault.vault_pii(PiiKind::Email, email);
                        let _ = arena.push(Span::new(start, end, SpanKind::Shadow(shadow)));
                    }
                }
                min_local_bound = end;
                i = end;
                continue;
            }
            min_local_bound = i + 1;
        }
        i += 1;
    }
}

fn scan_ips(src: &str, arena: &mut SpanArena<'_>, vault: &VaultHandle) {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        if (i == 0 || (!bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b':'))
            && bytes[i].is_ascii_hexdigit()
        {
            if bytes[i].is_ascii_digit() {
                if let Some(end) = match_ipv4(bytes, i) {
                    if !arena.overlaps_redacted(i, end) {
                        if let Some(ip_str) = src.get(i..end) {
                            let shadow = vault.vault_pii(PiiKind::Ip, ip_str);
                            let _ = arena.push(Span::new(i, end, SpanKind::Shadow(shadow)));
                        }
                    }
                    i = end;
                    continue;
                }
            }
            if let Some(end) = match_ipv6(bytes, i) {
                if !arena.overlaps_redacted(i, end) {
                    if let Some(ip_str) = src.get(i..end) {
                        let shadow = vault.vault_pii(PiiKind::Ip, ip_str);
                        let _ = arena.push(Span::new(i, end, SpanKind::Shadow(shadow)));
                    }
                }
                i = end;
                continue;
            }
        }
        i += 1;
    }
}

fn match_ipv4(bytes: &[u8], start: usize) -> Option<usize> {
    let len = bytes.len();
    let mut cursor = start;
    for octet_idx in 0..4 {
        let oct_start = cursor;
        let mut val: u16 = 0;
        while cursor < len && bytes[cursor].is_ascii_digit() && (cursor - oct_start) < 3 {
            val = val * 10 + u16::from(bytes[cursor] - b'0');
            cursor += 1;
        }
        if cursor == oct_start || val > 255 {
            return None;
        }
        if octet_idx < 3 {
            if cursor >= len || bytes[cursor] != b'.' {
                return None;
            }
            cursor += 1;
        }
    }
    if cursor < len && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'.') {
        return None;
    }
    Some(cursor)
}

fn match_ipv6(bytes: &[u8], start: usize) -> Option<usize> {
    let len = bytes.len();
    let mut cursor = start;
    let mut colon_count = 0usize;
    let mut group_count = 0usize;
    let mut group_len = 0usize;
    let mut double_colon_count = 0usize;

    while cursor < len {
        let b = bytes[cursor];
        if b.is_ascii_hexdigit() {
            if group_len == 0 {
                group_count += 1;
            }
            group_len += 1;
            if group_len > 4 {
                return None;
            }
            cursor += 1;
        } else if b == b':' {
            if cursor == start {
                return None;
            }
            if bytes[cursor - 1] == b':' {
                double_colon_count += 1;
                if double_colon_count > 1 {
                    return None;
                }
            }
            colon_count += 1;
            group_len = 0;
            cursor += 1;
        } else {
            break;
        }
    }

    if cursor > start + 1 && bytes[cursor - 1] == b':' && bytes[cursor - 2] != b':' {
        cursor -= 1;
        colon_count = colon_count.saturating_sub(1);
    }

    let valid_structure = (2..=7).contains(&colon_count)
        && ((double_colon_count == 1 && (2..=7).contains(&group_count))
            || (double_colon_count == 0 && group_count == 8));

    if valid_structure
        && cursor > start + 5
        && (cursor == len || !bytes[cursor].is_ascii_alphanumeric())
    {
        Some(cursor)
    } else {
        None
    }
}

fn scan_phones(src: &str, arena: &mut SpanArena<'_>, vault: &VaultHandle) {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        let is_start = (bytes[i] == b'+' && i + 1 < len && bytes[i + 1].is_ascii_digit())
            || (bytes[i] == b'(' && i + 4 < len && bytes[i + 1].is_ascii_digit())
            || (bytes[i].is_ascii_digit()
                && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
                && i + 3 < len
                && (bytes[i + 3] == b'-' || bytes[i + 3] == b' '));

        if is_start {
            let mut end = i;
            let mut digits = 0usize;
            let mut separators = 0usize;
            while end < len
                && (bytes[end].is_ascii_digit()
                    || matches!(bytes[end], b'+' | b'-' | b'(' | b')' | b' '))
            {
                if bytes[end].is_ascii_digit() {
                    digits += 1;
                } else if matches!(bytes[end], b'-' | b'(' | b')') {
                    separators += 1;
                }
                end += 1;
                if digits > 15 {
                    break;
                }
            }
            if digits > 15 {
                while end < len
                    && (bytes[end].is_ascii_digit()
                        || matches!(bytes[end], b'+' | b'-' | b'(' | b')' | b' '))
                {
                    end += 1;
                }
                i = end;
                continue;
            }
            while end > i && matches!(bytes[end - 1], b' ' | b'-' | b'(') {
                if matches!(bytes[end - 1], b'-' | b'(') {
                    separators = separators.saturating_sub(1);
                }
                end -= 1;
            }

            if (10..=15).contains(&digits)
                && (separators >= 2 || bytes[i] == b'+')
                && (end == len || !bytes[end].is_ascii_alphanumeric())
            {
                if !arena.overlaps_redacted(i, end) {
                    if let Some(phone_str) = src.get(i..end) {
                        let shadow = vault.vault_pii(PiiKind::Phone, phone_str);
                        let _ = arena.push(Span::new(i, end, SpanKind::Shadow(shadow)));
                    }
                }
                i = end;
                continue;
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_vendor_keys_and_key_value_suffixes_without_false_positives() {
        let payload = concat!(
            "aws=AKIAIOSFODNN7EXAMPLE ",
            "gh=ghp_1234567890abcdefghijklmnopqrstuv ",
            "slack=xoxb-123456789012-123456789012-abcdef ",
            "api_secret = \"my_super_secret_token_value_99\" ",
            "const KEY_COUNT = 5; let token_limit = 4096; commit=00deadbeef1234567890 ",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        );
        let mut arena = SpanArena::new(payload);
        scan_credentials(&mut arena);
        let out = arena.materialize();

        assert!(!out.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!out.contains("ghp_1234567890abcdefghijklmnopqrstuv"));
        assert!(!out.contains("xoxb-123456789012"));
        assert!(!out.contains("my_super_secret_token_value_99"));
        assert!(!out.contains("MIIEpAIBAAKCAQEA"));
        assert!(out
            .contains("const KEY_COUNT = 5; let token_limit = 4096; commit=00deadbeef1234567890 "));
        assert_eq!(out.matches(REDACTED_CREDENTIAL_MASK).count(), 5);
    }

    #[test]
    fn detects_emails_ipv4_ipv6_and_phones_without_false_ipv6() {
        let payload =
            "Contact alice@acme.com from 192.168.1.42 or 2001:db8:85a3::8a2e:370:7334 at +1-415-555-0199 (tag dead:beef:cafe).";
        let vault = VaultHandle::with_destinations(["crm_tool"]);
        let mut arena = SpanArena::new(payload);
        scan_and_vault_pii(&mut arena, &vault);
        let sanitized = arena.materialize();

        assert!(!sanitized.contains("alice@acme.com"));
        assert!(!sanitized.contains("192.168.1.42"));
        assert!(!sanitized.contains("2001:db8:85a3::8a2e:370:7334"));
        assert!(!sanitized.contains("+1-415-555-0199"));
        assert!(sanitized.contains("(tag dead:beef:cafe)"));
        assert!(sanitized.contains("<SHADOW:EMAIL_"));
        assert!(sanitized.contains("<SHADOW:IP_"));
        assert!(sanitized.contains("<SHADOW:PHONE_"));

        let restored = vault.rehydrate(&sanitized, "crm_tool");
        assert_eq!(restored, Ok(payload.to_owned()));
    }
}
