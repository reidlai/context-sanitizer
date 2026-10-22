//! Tier 1 (`L0`) deterministic secret redaction and operational PII shadow vaulting (`FR-1`, `FR-2`, `FR-3`, `AD-1`, `AD-3`).

pub mod entropy;
pub mod patterns;
pub mod vault;

use crate::arena::SpanArena;
pub use crate::guard::vault::{PiiKind, VaultHandle};

/// Canonical irreversible redaction token for root credentials (`FR-1`, `Consistency Conventions`).
pub const REDACTED_CREDENTIAL_MASK: &str = "[REDACTED_CREDENTIAL]";

/// Validates whether a byte is a valid local-part email character (`a-z`, `0-9`, `.`, `_`, `%`, `+`, `-`).
#[inline]
#[must_use]
pub const fn is_email_local_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'%' | b'+' | b'-')
}

/// Executes all Tier 1 (`L0`) read-only scanners over `arena` (vendor credentials, `*_KEY`/`*_SECRET`
/// assignments, standalone high-entropy secrets, and operational PII shadow vaulting).
pub fn run_tier1_guard(arena: &mut SpanArena<'_>, vault: &VaultHandle) {
    patterns::scan_credentials(arena);
    entropy::scan_high_entropy_tokens(arena);
    patterns::scan_and_vault_pii(arena, vault);
}

/// Sanitizes a raw string slice using Tier 1 (`L0`) deterministic credential redaction and PII
/// shadow vaulting with the supplied [`VaultHandle`].
#[must_use]
pub fn sanitize_str_with_vault(input: &str, vault: &VaultHandle) -> String {
    let mut arena = SpanArena::new(input);
    run_tier1_guard(&mut arena, vault);
    arena.materialize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier1_guard_redacts_secrets_and_vaults_pii_simultaneously() {
        let vault = VaultHandle::with_destinations(["db_lookup"]);
        let raw = "User alice@acme.com used AKIAIOSFODNN7EXAMPLE from 10.0.0.42";
        let clean = sanitize_str_with_vault(raw, &vault);
        assert!(clean.contains(REDACTED_CREDENTIAL_MASK));
        assert!(!clean.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(clean.contains("<SHADOW:EMAIL_"));
        assert!(clean.contains("<SHADOW:IP_"));
    }
}
