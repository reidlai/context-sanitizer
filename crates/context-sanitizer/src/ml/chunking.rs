//! 512-token sliding-window chunker with 128-token overlap and surprisal ranking (`FR-8`, `NFR-1`, `AD-5`).
//!
//! Prevents Neural Sponge CPU exhaustion DoS attacks by bounding neural evaluation to at most
//! top-$K$ ($K \le 2$) candidate windows ranked by Tier 2 Shannon self-information surprisal,
//! while eliminating boundary-splitting evasion attacks via a 128-token sliding stride.

use crate::compress::shannon::word_surprisal;

/// Default sliding window size in tokens.
pub const DEFAULT_WINDOW_SIZE: usize = 512;

/// Default sliding window overlap in tokens.
pub const DEFAULT_WINDOW_OVERLAP: usize = 128;

/// Default maximum number of top surprisal windows evaluated ($K \le 2$).
pub const DEFAULT_MAX_WINDOWS_K: usize = 2;

/// A candidate text window extracted from an input payload.
#[derive(Debug, Clone, PartialEq)]
pub struct TextWindow<'a> {
    /// Zero-based sequential window index in the original document.
    pub index: usize,
    /// Byte start offset in the original payload string.
    pub start_byte: usize,
    /// Byte end offset in the original payload string.
    pub end_byte: usize,
    /// Text slice corresponding to this window.
    pub text: &'a str,
    /// Number of tokens in this window.
    pub token_count: usize,
    /// Mean unigram Shannon surprisal score across tokens in this window.
    pub mean_surprisal: f32,
}

/// Tokenizes text into word / token byte spans `(start_byte, end_byte)`.
///
/// Fast $O(N)$ linear byte scanner without heap allocation per token.
#[must_use]
pub fn extract_token_spans(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut spans = Vec::with_capacity((len / 4).max(16));
    let mut i = 0;

    while i < len {
        // Skip whitespace
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        let start = i;
        // Non-whitespace sequence (token)
        while i < len && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let end = i;

        if start < end {
            spans.push((start, end));
        }
    }

    spans
}

/// Chunks `text` into 512-token sliding windows with 128-token overlap and ranks them
/// by mean Shannon surprisal, returning at most `max_k` highest-surprisal windows.
#[must_use]
pub fn chunk_and_rank_windows<'a>(
    text: &'a str,
    window_size: usize,
    overlap: usize,
    max_k: usize,
) -> Vec<TextWindow<'a>> {
    let text_len = text.len();
    if text_len == 0 {
        return Vec::new();
    }

    let effective_window_size = window_size.max(1);
    let effective_overlap = overlap.min(effective_window_size.saturating_sub(1));
    let effective_k = max_k.max(1);

    let token_spans = extract_token_spans(text);
    let total_tokens = token_spans.len();

    if total_tokens == 0 {
        return vec![TextWindow {
            index: 0,
            start_byte: 0,
            end_byte: text_len,
            text,
            token_count: 0,
            mean_surprisal: 0.0,
        }];
    }

    // Precalculate per-token surprisals
    let token_surprisals: Vec<f32> = token_spans
        .iter()
        .map(|&(start, end)| {
            if let Some(token_slice) = text.get(start..end) {
                word_surprisal(token_slice)
            } else {
                10.0
            }
        })
        .collect();

    // If total tokens fit in a single window, return single window directly
    if total_tokens <= effective_window_size {
        let total_surprisal: f32 = token_surprisals.iter().sum();
        let mean_surprisal = total_surprisal / (total_tokens as f32);
        return vec![TextWindow {
            index: 0,
            start_byte: 0,
            end_byte: text_len,
            text,
            token_count: total_tokens,
            mean_surprisal,
        }];
    }

    let stride = effective_window_size
        .saturating_sub(effective_overlap)
        .max(1);
    let mut windows = Vec::new();
    let mut token_start = 0;
    let mut win_idx = 0;

    while token_start < total_tokens {
        let token_end = (token_start + effective_window_size).min(total_tokens);
        let win_tokens = token_end - token_start;

        if let (Some(first_span), Some(last_span)) =
            (token_spans.get(token_start), token_spans.get(token_end - 1))
        {
            let start_byte = first_span.0;
            let end_byte = last_span.1.min(text_len);

            if let Some(window_slice) = text.get(start_byte..end_byte) {
                let window_surprisal_sum: f32 =
                    token_surprisals[token_start..token_end].iter().sum();
                let mean_surprisal = if win_tokens > 0 {
                    window_surprisal_sum / (win_tokens as f32)
                } else {
                    0.0
                };

                windows.push(TextWindow {
                    index: win_idx,
                    start_byte,
                    end_byte,
                    text: window_slice,
                    token_count: win_tokens,
                    mean_surprisal,
                });
            }
        }

        if token_end >= total_tokens {
            break;
        }

        token_start += stride;
        win_idx += 1;
    }

    // Rank windows in descending order of mean Shannon surprisal
    windows.sort_by(|a, b| b.mean_surprisal.total_cmp(&a.mean_surprisal));

    // Bound inference to at most top-K windows (Neural Sponge CPU exhaustion mitigation)
    windows.truncate(effective_k);

    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_short_payload_returns_one_window() {
        let input = "This is a brief conversational turn.";
        let windows = chunk_and_rank_windows(
            input,
            DEFAULT_WINDOW_SIZE,
            DEFAULT_WINDOW_OVERLAP,
            DEFAULT_MAX_WINDOWS_K,
        );
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].text, input);
        assert_eq!(windows[0].start_byte, 0);
        assert_eq!(windows[0].end_byte, input.len());
    }

    #[test]
    fn multi_window_sliding_stride_and_anti_sponge_bounding() {
        // Generate 1200 tokens (will generate multiple windows)
        let words: Vec<&str> = (0..1200)
            .map(|i| {
                if i % 10 == 0 {
                    "unusual_identifier"
                } else {
                    "the"
                }
            })
            .collect();
        let text = words.join(" ");

        let windows = chunk_and_rank_windows(
            &text, 512, 128, 2, // Bound to top 2
        );

        assert_eq!(windows.len(), 2, "Must be bounded to top-K = 2");
        assert!(windows[0].mean_surprisal > 0.0);
        assert!(windows[0].mean_surprisal >= windows[1].mean_surprisal);
    }

    #[test]
    fn overlap_covers_boundary_split_token_sequences() {
        // Create 600 words where an unusual word is located right around token 500
        let mut words = vec!["common"; 600];
        words[500] = "ADVERSARIAL_INJECTION_PAYLOAD";
        let text = words.join(" ");

        let windows = chunk_and_rank_windows(&text, 512, 128, 5);
        // Window 0 (0..512) and Window 1 (384..600) both overlap at token 500
        let contains_injection_count = windows
            .iter()
            .filter(|w| w.text.contains("ADVERSARIAL_INJECTION_PAYLOAD"))
            .count();
        assert!(
            contains_injection_count >= 1,
            "Boundary-split token sequence must be captured in overlapping window"
        );
    }
}
