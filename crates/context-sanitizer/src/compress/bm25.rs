//! Tail-query anchored Okapi BM25 relevance scorer (`FR-5`, `NFR-1`).
//!
//! Evaluates candidate historical tokens against the latest user query anchor to prevent
//! removing critical semantic entities or context needed by the downstream LLM.

use std::collections::BTreeMap;

/// Standard Okapi BM25 tuning constants.
pub const BM25_K1: f32 = 1.5;
pub const BM25_B: f32 = 0.75;

/// Compiled query terms and term frequencies for tail query anchoring.
#[derive(Debug, Clone, Default)]
pub struct TailQueryAnchor {
    terms: BTreeMap<String, usize>,
    total_terms: usize,
}

impl TailQueryAnchor {
    /// Builds a new [`TailQueryAnchor`] from the text of the latest user query.
    #[must_use]
    pub fn new(query_text: &str) -> Self {
        let mut terms = BTreeMap::new();
        let mut total_terms = 0usize;

        for word in query_text.split(|c: char| !c.is_alphanumeric()) {
            if word.is_empty() {
                continue;
            }
            let lower = word.to_ascii_lowercase();
            *terms.entry(lower).or_insert(0) += 1;
            total_terms += 1;
        }

        Self { terms, total_terms }
    }

    /// Returns `true` if the anchor contains no query terms.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Evaluates the BM25 relevance score of a candidate word against this query anchor.
    ///
    /// Tokens appearing in the user query receive a positive relevance score proportional
    /// to term frequency, while unrelated words return `0.0`.
    #[must_use]
    pub fn score_term(&self, word: &str) -> f32 {
        if self.terms.is_empty() {
            return 0.0;
        }

        let lower = word.to_ascii_lowercase();
        if let Some(&tf) = self.terms.get(&lower) {
            let tf_f = tf as f32;
            // IDF approximation for single-query context with default background frequency
            let idf = ((self.total_terms as f32 + 0.5) / (tf_f + 0.5) + 1.0).ln();
            let numerator = tf_f * (BM25_K1 + 1.0);
            let denominator = tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B);
            idf * (numerator / denominator)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_terms_receive_positive_scores_and_unrelated_zero() {
        let query = "Where is the database password stored?";
        let anchor = TailQueryAnchor::new(query);

        assert!(anchor.score_term("database") > 0.0);
        assert!(anchor.score_term("password") > 0.0);
        assert!(anchor.score_term("stored") > 0.0);
        assert_eq!(anchor.score_term("unrelated_token"), 0.0);
        assert_eq!(anchor.score_term("weather"), 0.0);
    }
}
