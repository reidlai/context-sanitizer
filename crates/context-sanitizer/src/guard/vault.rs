//! Session-scoped thread-safe PII Shadow Vault and destination-whitelisted rehydration (`FR-2`, `FR-3`, `AD-3`).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::SanitizerError;

static FALLBACK_SALT_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Supported operational PII categories for session-scoped shadow vaulting (`FR-2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiKind {
    /// Email address (`<SHADOW:EMAIL_<HEX6>>`).
    Email,
    /// IPv4 or IPv6 address (`<SHADOW:IP_<HEX6>>`).
    Ip,
    /// Telephone number (`<SHADOW:PHONE_<HEX6>>`).
    Phone,
}

impl PiiKind {
    /// Returns the canonical uppercase prefix used inside `<SHADOW:<TYPE>_<HEX6>>`.
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Email => "EMAIL",
            Self::Ip => "IP",
            Self::Phone => "PHONE",
        }
    }
}

#[derive(Debug)]
struct VaultInner {
    forward: HashMap<(PiiKind, String), String>,
    reverse: HashMap<String, String>,
    whitelist: HashSet<String>,
    salt: [u8; 16],
}

impl Drop for VaultInner {
    fn drop(&mut self) {
        self.forward.clear();
        self.reverse.clear();
        self.whitelist.clear();
        self.salt.fill(0);
    }
}

/// Thread-safe, session-scoped handle mapping raw operational PII to `<SHADOW:<TYPE>_<HEX6>>` tokens (`AD-3`).
#[derive(Debug, Clone)]
pub struct VaultHandle {
    inner: Arc<RwLock<VaultInner>>,
}

impl Default for VaultHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl VaultHandle {
    /// Creates a new session-isolated [`VaultHandle`] with a fresh 128-bit cryptographic random salt.
    #[must_use]
    pub fn new() -> Self {
        Self::with_destinations(std::iter::empty::<&str>())
    }

    /// Creates a new session-isolated [`VaultHandle`] pre-configured with authorized tool destinations.
    #[must_use]
    pub fn with_destinations<I, S>(destinations: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut salt = [0u8; 16];
        if getrandom::getrandom(&mut salt).is_err() {
            let seq = FALLBACK_SALT_COUNTER.fetch_add(1, Ordering::Relaxed);
            let seq_bytes = seq.to_le_bytes();
            for (i, byte) in salt.iter_mut().enumerate() {
                let mix = seq_bytes[i % 8]
                    .wrapping_mul(37)
                    .wrapping_add(u8::try_from(i).unwrap_or(0));
                *byte = mix ^ 0xA5;
            }
        }

        let whitelist = destinations.into_iter().map(Into::into).collect();
        Self {
            inner: Arc::new(RwLock::new(VaultInner {
                forward: HashMap::new(),
                reverse: HashMap::new(),
                whitelist,
                salt,
            })),
        }
    }

    /// Creates a [`VaultHandle`] with an explicit 128-bit session salt (useful for deterministic testing).
    #[must_use]
    pub fn with_salt_and_destinations<I, S>(salt: [u8; 16], destinations: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let whitelist = destinations.into_iter().map(Into::into).collect();
        Self {
            inner: Arc::new(RwLock::new(VaultInner {
                forward: HashMap::new(),
                reverse: HashMap::new(),
                whitelist,
                salt,
            })),
        }
    }

    /// Registers an additional authorized tool destination for local `.rehydrate()`.
    pub fn allow_destination(&self, destination: impl Into<String>) {
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.whitelist.insert(destination.into());
    }

    /// Returns the number of unique PII entities vaulted in this session.
    #[must_use]
    pub fn len(&self) -> usize {
        let guard = self
            .inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.forward.len()
    }

    /// Returns `true` if no PII entities have been vaulted in this session.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vaults a raw operational PII value and returns its deterministic session-scoped shadow token
    /// (`<SHADOW:<TYPE>_<HEX6>>`).
    #[must_use]
    pub fn vault_pii(&self, kind: PiiKind, raw_pii: &str) -> String {
        let key = (kind, raw_pii.to_owned());
        {
            let guard = self
                .inner
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(existing) = guard.forward.get(&key) {
                return existing.clone();
            }
        }

        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = guard.forward.get(&key) {
            return existing.clone();
        }

        let mut attempt = 0u32;
        let token = loop {
            let hash24 = compute_salted_hex6(&guard.salt, kind.as_str(), raw_pii, attempt);
            let candidate = format!("<SHADOW:{}_{hash24:06x}>", kind.as_str());
            match guard.reverse.get(&candidate) {
                Some(existing_raw) if existing_raw != raw_pii => {
                    attempt = attempt.saturating_add(1);
                }
                _ => break candidate,
            }
        };

        guard.forward.insert(key, token.clone());
        guard.reverse.insert(token.clone(), raw_pii.to_owned());
        token
    }

    /// Rehydrates all `<SHADOW:<TYPE>_<HEX6>>` tokens present in `tool_args` back to their raw PII
    /// values if and only if `destination` is registered in the session's authorized tool whitelist (`FR-3`).
    ///
    /// # Errors
    /// Returns [`SanitizerError::UnauthorizedRehydration`] if `destination` is not whitelisted.
    pub fn rehydrate(&self, tool_args: &str, destination: &str) -> Result<String, SanitizerError> {
        let guard = self
            .inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if !guard.whitelist.contains(destination) {
            return Err(SanitizerError::UnauthorizedRehydration(
                destination.to_owned(),
            ));
        }

        if guard.reverse.is_empty() || !tool_args.contains("<SHADOW:") {
            return Ok(tool_args.to_owned());
        }

        let mut out = String::with_capacity(tool_args.len());
        let mut cursor = 0usize;
        while let Some(rel_start) = tool_args.get(cursor..).and_then(|s| s.find("<SHADOW:")) {
            let token_start = cursor + rel_start;
            if let Some(prefix) = tool_args.get(cursor..token_start) {
                out.push_str(prefix);
            }

            let after_prefix = token_start + "<SHADOW:".len();
            if let Some(rel_end) = tool_args.get(after_prefix..).and_then(|s| s.find('>')) {
                let token_end = after_prefix + rel_end + 1;
                // Check if another `<SHADOW:` starts before `token_end` (unclosed outer `<SHADOW:` prefix)
                if let Some(nested_rel) = tool_args
                    .get(after_prefix..token_end)
                    .and_then(|s| s.find("<SHADOW:"))
                {
                    let nested_start = after_prefix + nested_rel;
                    if let Some(skipped) = tool_args.get(token_start..nested_start) {
                        out.push_str(skipped);
                    }
                    cursor = nested_start;
                    continue;
                }

                if let Some(candidate) = tool_args.get(token_start..token_end) {
                    if let Some(raw_value) = guard.reverse.get(candidate) {
                        out.push_str(raw_value);
                    } else {
                        out.push_str(candidate);
                    }
                }
                cursor = token_end;
            } else {
                if let Some(rest) = tool_args.get(token_start..) {
                    out.push_str(rest);
                }
                cursor = tool_args.len();
                break;
            }
        }
        if cursor < tool_args.len() {
            if let Some(tail) = tool_args.get(cursor..) {
                out.push_str(tail);
            }
        }

        Ok(out)
    }
}

/// Keyed 64-bit SipHash-style / FNV-1a avalanche mixer producing a 24-bit (`<HEX6>`) digest.
#[inline]
fn compute_salted_hex6(salt: &[u8; 16], kind: &str, raw: &str, attempt: u32) -> u32 {
    let mut state: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in salt
        .iter()
        .chain(kind.as_bytes())
        .chain(&[0xFF])
        .chain(raw.as_bytes())
        .chain(&attempt.to_le_bytes())
    {
        state ^= u64::from(b);
        state = state.wrapping_mul(0x0100_0000_01b3);
        state ^= state >> 32;
    }
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    state = state.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    state ^= state >> 33;
    u32::try_from(state & 0x00FF_FFFF).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_session_deterministic_and_cross_session_isolated() {
        let vault_a = VaultHandle::with_destinations(["db_lookup"]);
        let vault_b = VaultHandle::with_destinations(["db_lookup"]);

        let t1 = vault_a.vault_pii(PiiKind::Email, "alice@acme.com");
        let t2 = vault_a.vault_pii(PiiKind::Email, "alice@acme.com");
        assert_eq!(t1, t2);
        assert!(t1.starts_with("<SHADOW:EMAIL_"));
        assert!(t1.ends_with('>'));
        assert_eq!(t1.len(), "<SHADOW:EMAIL_8f3a2c>".len());
        assert_eq!(vault_a.len(), 1);
        assert!(!vault_a.is_empty());

        let t_other = vault_b.vault_pii(PiiKind::Email, "alice@acme.com");
        assert_ne!(t1, t_other);
    }

    #[test]
    fn whitelisted_rehydration_allows_registered_and_rejects_unregistered() {
        let vault = VaultHandle::new();
        assert!(vault.is_empty());
        vault.allow_destination("stripe_lookup");

        let email_tok = vault.vault_pii(PiiKind::Email, "alice@acme.com");
        let ip_tok = vault.vault_pii(PiiKind::Ip, "192.168.1.42");
        let args = format!(
            r#"{{"email":"<SHADOW:broken_{email_tok}","ip":"{ip_tok}","unknown":"<SHADOW:EMAIL_000000>"}}"#
        );

        let rehydrated = vault.rehydrate(&args, "stripe_lookup");
        assert_eq!(
            rehydrated,
            Ok(
                r#"{"email":"<SHADOW:broken_alice@acme.com","ip":"192.168.1.42","unknown":"<SHADOW:EMAIL_000000>"}"#
                    .to_owned()
            )
        );

        let rejected = vault.rehydrate(&args, "https://evil.com/webhook");
        assert_eq!(
            rejected,
            Err(SanitizerError::UnauthorizedRehydration(
                "https://evil.com/webhook".to_owned()
            ))
        );
    }
}
