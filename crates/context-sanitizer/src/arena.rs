//! Zero-copy span-annotated buffer arena and single-allocation materializer (`AD-1`).
//!
//! All scanners execute as read-only passes over an immutable `&str` buffer, emitting [`Span`]
//! annotations into a [`SpanArena`]. When [`SpanArena::materialize`] is invoked, overlapping spans
//! are resolved according to the strict precedence order:
//! `Redacted` > `Shadow` > `Locked` > `Pruned` > `Retained`.

use crate::SanitizerError;

/// Precedence-ordered span classifications (`Redacted` > `Shadow` > `Locked` > `Pruned` > `Retained`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanKind {
    /// Irreversible one-way credential mask (`[REDACTED_CREDENTIAL]`).
    Redacted(&'static str),
    /// Session-scoped operational PII shadow token (`<SHADOW:<TYPE>_<HEX6>>`).
    Shadow(String),
    /// Immutable syntax token (JSON structural delimiter, code fence, shell substitution).
    Locked,
    /// Low-entropy token marked for removal during Tier 2 compression.
    Pruned,
    /// Token retained in the final materialized payload.
    Retained,
}

impl SpanKind {
    /// Higher numeric rank means higher priority during overlap resolution (`AD-1`).
    #[inline]
    #[must_use]
    pub const fn precedence(&self) -> u8 {
        match self {
            Self::Redacted(_) => 5,
            Self::Shadow(_) => 4,
            Self::Locked => 3,
            Self::Pruned => 2,
            Self::Retained => 1,
        }
    }
}

/// Non-overlapping or overlapping byte span annotation over an immutable input payload (`AD-1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// Inclusive start byte offset (UTF-8 char boundary).
    pub start: usize,
    /// Exclusive end byte offset (UTF-8 char boundary).
    pub end: usize,
    /// Classification and precedence of this span.
    pub kind: SpanKind,
}

impl Span {
    /// Constructs a new [`Span`].
    #[inline]
    #[must_use]
    pub const fn new(start: usize, end: usize, kind: SpanKind) -> Self {
        Self { start, end, kind }
    }
}

/// Zero-copy span arena holding annotations over an immutable source `&'a str` buffer.
#[derive(Debug, Clone)]
pub struct SpanArena<'a> {
    source: &'a str,
    spans: Vec<Span>,
}

impl<'a> SpanArena<'a> {
    /// Creates a new [`SpanArena`] bound to `source`.
    #[inline]
    #[must_use]
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            spans: Vec::new(),
        }
    }

    /// Creates a new [`SpanArena`] with pre-allocated span capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(source: &'a str, capacity: usize) -> Self {
        Self {
            source,
            spans: Vec::with_capacity(capacity),
        }
    }

    /// Returns the underlying immutable source slice.
    #[inline]
    #[must_use]
    pub const fn source(&self) -> &'a str {
        self.source
    }

    /// Returns all registered spans in insertion order.
    #[inline]
    #[must_use]
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Returns `true` if the byte range `start..end` overlaps any existing `SpanKind::Redacted` span.
    #[inline]
    #[must_use]
    pub fn overlaps_redacted(&self, start: usize, end: usize) -> bool {
        self.spans
            .iter()
            .any(|s| matches!(s.kind, SpanKind::Redacted(_)) && start < s.end && end > s.start)
    }

    /// Pushes a [`Span`] into the arena after validating that `start <= end`, `end <= source.len()`,
    /// and both `start` and `end` lie on valid UTF-8 character boundaries.
    ///
    /// # Errors
    /// Returns [`SanitizerError::InvalidSpanBounds`] if offsets are out of range or split a multi-byte UTF-8 character.
    pub fn push(&mut self, span: Span) -> Result<(), SanitizerError> {
        let len = self.source.len();
        if span.start > span.end
            || span.end > len
            || !self.source.is_char_boundary(span.start)
            || !self.source.is_char_boundary(span.end)
        {
            return Err(SanitizerError::InvalidSpanBounds {
                start: span.start,
                end: span.end,
                len,
            });
        }
        if span.start < span.end {
            self.spans.push(span);
        }
        Ok(())
    }

    /// Resolves overlapping spans by strict precedence (`Redacted` > `Shadow` > `Locked` > `Pruned` > `Retained`)
    /// and materializes the transformed string in a single final allocation (`AD-1`).
    #[must_use]
    pub fn materialize(&self) -> String {
        self.materialize_internal(true)
    }

    /// Materializes the payload with `Redacted` and `Shadow` substitutions applied, but with all
    /// `Pruned` spans discarded (`AD-4`). Used for fail-open recovery when syntax validation fails.
    #[must_use]
    pub fn materialize_without_pruning(&self) -> String {
        self.materialize_internal(false)
    }

    /// Alias for [`Self::materialize_without_pruning`] enforcing the Fail-Open Syntax Invariant (`AD-4`).
    #[must_use]
    pub fn materialize_fail_open(&self) -> String {
        self.materialize_without_pruning()
    }

    fn materialize_internal(&self, apply_pruning: bool) -> String {
        if self.spans.is_empty() {
            return self.source.to_owned();
        }

        // 1. Resolve `Redacted` (precedence 5) first. Overlapping `Redacted` spans must merge into
        //    the union of their byte ranges (Epic 1 Retro Action Item 1) so no credential substring leaks.
        let mut redacted_spans: Vec<Span> = self
            .spans
            .iter()
            .filter(|s| matches!(s.kind, SpanKind::Redacted(_)))
            .cloned()
            .collect();
        redacted_spans.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));

        let mut accepted: Vec<Span> = Vec::with_capacity(self.spans.len());
        for span in redacted_spans {
            let mask = match span.kind {
                SpanKind::Redacted(m) => m,
                _ => continue,
            };
            if let Some(last) = accepted.last_mut() {
                if span.start <= last.end {
                    last.end = last.end.max(span.end);
                    continue;
                }
            }
            accepted.push(Span::new(span.start, span.end, SpanKind::Redacted(mask)));
        }

        // 2. Resolve `Shadow` (precedence 4) replacements. Shadow spans that overlap any accepted
        //    `Redacted` union or prior `Shadow` span are dropped.
        let mut shadow_candidates: Vec<&Span> = self
            .spans
            .iter()
            .filter(|s| matches!(s.kind, SpanKind::Shadow(_)))
            .collect();
        shadow_candidates.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
        });

        for candidate in shadow_candidates {
            let overlaps = accepted
                .iter()
                .any(|existing| candidate.start < existing.end && candidate.end > existing.start);
            if !overlaps {
                accepted.push(candidate.clone());
            }
        }

        if apply_pruning {
            let mut prune_intervals: Vec<(usize, usize)> = self
                .spans
                .iter()
                .filter(|s| matches!(s.kind, SpanKind::Pruned))
                .map(|s| (s.start, s.end))
                .collect();

            if !prune_intervals.is_empty() {
                // Sort accepted higher-priority spans by start offset for binary-search overlap checks
                accepted.sort_by_key(|s| s.start);

                // Collect & merge `Locked` protection ranges (precedence 3) which block `Pruned` (precedence 2) spans
                let mut raw_locked: Vec<(usize, usize)> = self
                    .spans
                    .iter()
                    .filter(|s| matches!(s.kind, SpanKind::Locked))
                    .map(|s| (s.start, s.end))
                    .collect();
                raw_locked.sort_unstable_by_key(|&(start, _)| start);

                let mut locked_ranges: Vec<(usize, usize)> = Vec::with_capacity(raw_locked.len());
                for (start, end) in raw_locked {
                    if let Some(last) = locked_ranges.last_mut() {
                        if start <= last.1 {
                            last.1 = last.1.max(end);
                            continue;
                        }
                    }
                    locked_ranges.push((start, end));
                }

                // Merge overlapping/adjacent `Pruned` spans (precedence 2) and accept those that do not
                // overlap any `Redacted`, `Shadow`, or `Locked` span.
                prune_intervals.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));

                let mut merged_prunes: Vec<(usize, usize)> =
                    Vec::with_capacity(prune_intervals.len());
                for (start, end) in prune_intervals {
                    let acc_idx = accepted.partition_point(|ex| ex.end <= start);
                    let overlaps_accepted =
                        acc_idx < accepted.len() && accepted[acc_idx].start < end;

                    let lock_idx = locked_ranges.partition_point(|&(_, l_end)| l_end <= start);
                    let overlaps_locked =
                        lock_idx < locked_ranges.len() && locked_ranges[lock_idx].0 < end;

                    let overlaps_higher = overlaps_accepted || overlaps_locked;
                    if overlaps_higher {
                        continue;
                    }
                    if let Some(last) = merged_prunes.last_mut() {
                        if start <= last.1 {
                            last.1 = last.1.max(end);
                            continue;
                        }
                    }
                    merged_prunes.push((start, end));
                }

                for (start, end) in merged_prunes {
                    accepted.push(Span::new(start, end, SpanKind::Pruned));
                }
            }
        }

        // Sort accepted non-overlapping spans by byte start offset.
        accepted.sort_by_key(|s| s.start);

        // Calculate exact output byte length for a single allocation.
        let mut output_len = self.source.len();
        for span in &accepted {
            let span_len = span.end - span.start;
            match &span.kind {
                SpanKind::Redacted(mask) => {
                    output_len = output_len.saturating_sub(span_len) + mask.len();
                }
                SpanKind::Shadow(token) => {
                    output_len = output_len.saturating_sub(span_len) + token.len();
                }
                SpanKind::Pruned => {
                    output_len = output_len.saturating_sub(span_len);
                }
                SpanKind::Locked | SpanKind::Retained => {}
            }
        }

        let mut out = String::with_capacity(output_len);
        let mut cursor = 0usize;
        for span in &accepted {
            if cursor < span.start {
                if let Some(prefix) = self.source.get(cursor..span.start) {
                    out.push_str(prefix);
                }
            }
            match &span.kind {
                SpanKind::Redacted(mask) => out.push_str(mask),
                SpanKind::Shadow(token) => out.push_str(token),
                SpanKind::Pruned => {}
                SpanKind::Locked | SpanKind::Retained => {
                    if let Some(slice) = self.source.get(span.start..span.end) {
                        out.push_str(slice);
                    }
                }
            }
            cursor = span.end;
        }
        if cursor < self.source.len() {
            if let Some(tail) = self.source.get(cursor..) {
                out.push_str(tail);
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_arena_returns_source_verbatim() {
        let src = "Hello, world! 🦀";
        let arena = SpanArena::new(src);
        assert_eq!(arena.source(), src);
        assert!(arena.spans().is_empty());
        assert_eq!(arena.materialize(), src);
    }

    #[test]
    fn redacted_and_shadow_override_locked_and_pruned() {
        let src = "```bash\nexport AWS_KEY=AKIAIOSFODNN7EXAMPLE\n# alice@acme.com\n```";
        let mut arena = SpanArena::with_capacity(src, 6);
        // Lock the entire code fence block (0..src.len())
        assert!(arena
            .push(Span::new(0, src.len(), SpanKind::Locked))
            .is_ok());
        // Attempt to prune inside locked block (should be blocked by Locked)
        assert!(arena.push(Span::new(8, 14, SpanKind::Pruned)).is_ok());
        // Push a Shadow span on the AWS key range AND an overlapping Redacted span:
        // Redacted (rank 5) MUST override Shadow (rank 4).
        assert!(arena
            .push(Span::new(
                23,
                43,
                SpanKind::Shadow("<SHADOW:EMAIL_000000>".to_owned())
            ))
            .is_ok());
        assert!(arena
            .push(Span::new(
                23,
                43,
                SpanKind::Redacted("[REDACTED_CREDENTIAL]")
            ))
            .is_ok());
        // Shadow the email inside the locked block (must override Locked!)
        assert!(arena
            .push(Span::new(
                46,
                60,
                SpanKind::Shadow("<SHADOW:EMAIL_8f3a2c>".to_owned())
            ))
            .is_ok());

        let result = arena.materialize();
        assert_eq!(
            result,
            "```bash\nexport AWS_KEY=[REDACTED_CREDENTIAL]\n# <SHADOW:EMAIL_8f3a2c>\n```"
        );
    }

    #[test]
    fn pruned_overrides_retained_merges_overlapping_prunes_and_respects_locked() {
        let src = "keep_this drop_this lock_this";
        let mut arena = SpanArena::new(src);
        assert!(arena.push(Span::new(0, 9, SpanKind::Retained)).is_ok());
        // Overlapping Pruned spans (9..14 and 12..19) merge into 9..19 and override Retained (rank 1)
        assert!(arena.push(Span::new(9, 19, SpanKind::Retained)).is_ok());
        assert!(arena.push(Span::new(9, 14, SpanKind::Pruned)).is_ok());
        assert!(arena.push(Span::new(12, 19, SpanKind::Pruned)).is_ok());
        assert!(arena.push(Span::new(20, 29, SpanKind::Locked)).is_ok());
        // Overlapping prune on locked span is ignored
        assert!(arena.push(Span::new(20, 29, SpanKind::Pruned)).is_ok());

        assert_eq!(arena.materialize(), "keep_this lock_this");
    }

    #[test]
    fn rejects_invalid_utf8_boundaries_and_out_of_bounds() {
        let src = "🦀rust"; // '🦀' is 4 bytes (0..4)
        let mut arena = SpanArena::new(src);
        assert_eq!(
            arena.push(Span::new(1, 3, SpanKind::Pruned)),
            Err(SanitizerError::InvalidSpanBounds {
                start: 1,
                end: 3,
                len: src.len()
            })
        );
        assert_eq!(
            arena.push(Span::new(5, 4, SpanKind::Pruned)),
            Err(SanitizerError::InvalidSpanBounds {
                start: 5,
                end: 4,
                len: src.len()
            })
        );
        assert_eq!(
            arena.push(Span::new(0, 99, SpanKind::Pruned)),
            Err(SanitizerError::InvalidSpanBounds {
                start: 0,
                end: 99,
                len: src.len()
            })
        );
        // Zero-length span on valid boundary is a no-op
        assert!(arena.push(Span::new(4, 4, SpanKind::Retained)).is_ok());
        assert!(arena.spans().is_empty());
    }

    #[test]
    fn overlapping_redacted_spans_merge_into_union() {
        let src = "prefix_SECRET_SUBPART_EXTRA_suffix";
        let mut arena = SpanArena::new(src);
        // Span 1: "SECRET_SUBPART" (7..21)
        assert!(arena
            .push(Span::new(
                7,
                21,
                SpanKind::Redacted("[REDACTED_CREDENTIAL]")
            ))
            .is_ok());
        // Span 2: "SUBPART_EXTRA" (14..27) overlapping with span 1
        assert!(arena
            .push(Span::new(
                14,
                27,
                SpanKind::Redacted("[REDACTED_CREDENTIAL]")
            ))
            .is_ok());

        assert_eq!(arena.materialize(), "prefix_[REDACTED_CREDENTIAL]_suffix");
    }

    #[test]
    fn materialize_without_pruning_discards_pruned_spans_and_keeps_redactions() {
        let src = "hello SECRET_TOKEN drop_me world";
        let mut arena = SpanArena::new(src);
        assert!(arena
            .push(Span::new(6, 18, SpanKind::Redacted("[REDACTED]")))
            .is_ok());
        assert!(arena.push(Span::new(19, 27, SpanKind::Pruned)).is_ok());

        // With pruning:
        assert_eq!(arena.materialize(), "hello [REDACTED] world");
        // Without pruning (fail-open):
        assert_eq!(
            arena.materialize_without_pruning(),
            "hello [REDACTED] drop_me world"
        );
        assert_eq!(
            arena.materialize_fail_open(),
            "hello [REDACTED] drop_me world"
        );
    }
}
