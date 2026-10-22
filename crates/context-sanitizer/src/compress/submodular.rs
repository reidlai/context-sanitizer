//! $O(N \log N)$ lazy-greedy max-heap monotone submodular $(1 - 1/e)$ knapsack optimizer (`FR-5`, `NFR-1`).
//!
//! Selects low-significance filler tokens to mark as [`SpanKind::Pruned`] while strictly respecting
//! immutable [`SpanKind::Locked`] syntax spans, Tier 1 secrets, and auto-anchored [`SpanKind::Retained`] queries.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::arena::{Span, SpanArena, SpanKind};
use crate::compress::bm25::TailQueryAnchor;
use crate::compress::shannon::word_surprisal;
use crate::SanitizerError;

/// Knapsack item wrapper for lazy-greedy max-heap evaluation.
#[derive(Debug, Clone, PartialEq)]
struct HeapItem {
    id: usize,
    priority: f32,
    seen_pruned_count: u32,
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.total_cmp(&other.priority)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct CandidateSpan<'a> {
    start: usize,
    end: usize,
    len: usize,
    word: &'a str,
    base_significance: f32,
}

/// Applies lazy-greedy submodular knapsack selection to mark pruneable tokens as [`SpanKind::Pruned`].
///
/// - `target_rate`: Target compression ratio ($0.0 \le \text{rate} \le 0.70$).
/// - `tail_query`: Optional user query text for BM25 tail anchoring.
///
/// # Errors
/// Returns [`SanitizerError::InvalidSpanBounds`] if any selected span bounds violate UTF-8 boundaries.
pub fn prune_unlocked_context(
    arena: &mut SpanArena<'_>,
    target_rate: f32,
    tail_query: Option<&str>,
) -> Result<usize, SanitizerError> {
    let source = arena.source();
    let total_len = source.len();
    if total_len == 0 {
        return Ok(0);
    }

    let clamped_rate = target_rate.clamp(0.0, 0.70);
    let anchor = tail_query.map(TailQueryAnchor::new).unwrap_or_default();

    // 1. Collect and merge all existing protected byte ranges (Redacted, Shadow, Locked, Retained).
    let mut raw_protected: Vec<(usize, usize)> = arena
        .spans()
        .iter()
        .filter(|s| !matches!(s.kind, SpanKind::Pruned))
        .map(|s| (s.start, s.end))
        .collect();
    raw_protected.sort_unstable_by_key(|&(start, _)| start);

    let mut protected_ranges: Vec<(usize, usize)> = Vec::with_capacity(raw_protected.len());
    for (start, end) in raw_protected {
        if let Some(last) = protected_ranges.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        protected_ranges.push((start, end));
    }

    // 2. Identify pruneable candidate word tokens outside protected ranges in linear O(N) time.
    let mut candidates: Vec<CandidateSpan<'_>> = Vec::with_capacity((total_len / 8).max(16));
    let bytes = source.as_bytes();
    let mut i = 0usize;
    let mut p_idx = 0usize;
    let num_protected = protected_ranges.len();

    while i < total_len {
        // Fast-forward protected index past current cursor
        while p_idx < num_protected && protected_ranges[p_idx].1 <= i {
            p_idx += 1;
        }

        // Jump over protected range if cursor falls inside it
        if p_idx < num_protected && i >= protected_ranges[p_idx].0 {
            i = protected_ranges[p_idx].1;
            p_idx += 1;
            continue;
        }

        // Skip leading whitespace / punctuation
        while i < total_len {
            if p_idx < num_protected && i >= protected_ranges[p_idx].0 {
                break;
            }
            if !bytes[i].is_ascii_whitespace() && !bytes[i].is_ascii_punctuation() {
                break;
            }
            i += 1;
        }
        if i >= total_len {
            break;
        }
        if p_idx < num_protected && i >= protected_ranges[p_idx].0 {
            continue;
        }

        let start = i;
        while i < total_len && !bytes[i].is_ascii_whitespace() && !bytes[i].is_ascii_punctuation() {
            if p_idx < num_protected && i >= protected_ranges[p_idx].0 {
                break;
            }
            i += 1;
        }
        let end = i;

        let mut prune_end = end;
        if prune_end < total_len && bytes[prune_end] == b' ' {
            let next_is_protected =
                p_idx < num_protected && (prune_end + 1) > protected_ranges[p_idx].0;
            if !next_is_protected {
                prune_end += 1;
            }
        }

        if start < prune_end {
            if let Some(word_slice) = source.get(start..end) {
                let b_score = anchor.score_term(word_slice);
                if b_score > 0.0 {
                    continue;
                }

                let s_score = word_surprisal(word_slice);
                candidates.push(CandidateSpan {
                    start,
                    end: prune_end,
                    len: prune_end - start,
                    word: word_slice,
                    base_significance: s_score,
                });
            }
        }
    }

    if candidates.is_empty() {
        return Ok(0);
    }

    let candidate_total_bytes: usize = candidates.iter().map(|c| c.len).sum();
    let target_prune_bytes = ((candidate_total_bytes as f32) * clamped_rate) as usize;
    if target_prune_bytes == 0 {
        return Ok(0);
    }

    // 3. Initialize lazy-greedy max-heap ordered by marginal priority:
    // priority = len / (base_significance + 0.1)
    let mut heap = BinaryHeap::with_capacity(candidates.len());
    for (id, candidate) in candidates.iter().enumerate() {
        let priority = (candidate.len as f32) / (candidate.base_significance + 0.1);
        heap.push(HeapItem {
            id,
            priority,
            seen_pruned_count: 0,
        });
    }

    let mut pruned_bytes = 0usize;
    let mut word_pruned_counts: HashMap<&str, u32> =
        HashMap::with_capacity(candidates.len().min(1024));
    let mut selected_candidates = Vec::with_capacity(candidates.len().min(512));

    // 4. Lazy-greedy execution loop
    while let Some(top) = heap.pop() {
        if pruned_bytes >= target_prune_bytes {
            break;
        }

        let cand = &candidates[top.id];
        let current_word_pruned = word_pruned_counts.get(cand.word).copied().unwrap_or(0);

        if top.seen_pruned_count == current_word_pruned {
            // Fresh priority: greedily accept
            selected_candidates.push(top.id);
            pruned_bytes += cand.len;
            *word_pruned_counts.entry(cand.word).or_insert(0) += 1;
        } else {
            // Recompute marginal priority with diminishing returns on repeated words
            let penalty = 1.0 + 0.25 * (current_word_pruned as f32);
            let recomputed_priority =
                (cand.len as f32) / ((cand.base_significance * penalty) + 0.1);

            let is_still_max = match heap.peek() {
                Some(next_top) => recomputed_priority >= next_top.priority,
                None => true,
            };

            if is_still_max {
                selected_candidates.push(top.id);
                pruned_bytes += cand.len;
                *word_pruned_counts.entry(cand.word).or_insert(0) += 1;
            } else {
                heap.push(HeapItem {
                    id: top.id,
                    priority: recomputed_priority,
                    seen_pruned_count: current_word_pruned,
                });
            }
        }
    }

    // 5. Register selected spans in SpanArena
    let mut count = 0usize;
    for id in selected_candidates {
        let cand = &candidates[id];
        arena.push(Span::new(cand.start, cand.end, SpanKind::Pruned))?;
        count += 1;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submodular_pruning_targets_filler_and_preserves_query_entities() {
        let text = "The user asked for database configuration details and the credentials for the database";
        let mut arena = SpanArena::new(text);

        let pruned_count =
            prune_unlocked_context(&mut arena, 0.50, Some("database credentials")).unwrap_or(0);
        assert!(pruned_count > 0);

        let result = arena.materialize();
        // Query entities must be preserved
        assert!(result.contains("database"));
        assert!(result.contains("credentials"));
    }
}
