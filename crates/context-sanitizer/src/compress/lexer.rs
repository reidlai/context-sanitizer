//! Pure-Rust lexical syntax token locking and fail-open validation (`AD-1`, `AD-4`, `FR-4`).
//!
//! Uses a `logos` DFA tokenizer to tag structural JSON delimiters, JSON keys, markdown code fences,
//! inline shell commands, subshells, and `$VAR` environment variable expansions as [`SpanKind::Locked`].

use logos::Logos;

use crate::arena::{Span, SpanArena, SpanKind};
use crate::SanitizerError;

/// Lexical tokens scanned for syntax locking (`AD-1`, `FR-4`).
#[derive(Logos, Debug, PartialEq, Eq, Clone, Copy)]
pub enum SyntaxToken {
    /// Markdown code fences: triple backticks or tildes with optional language tag (` ```bash\n `, ` ~~~ `).
    #[regex(r"```[a-zA-Z0-9_.-]*\r?\n?|```|~~~[a-zA-Z0-9_.-]*\r?\n?|~~~")]
    MarkdownFence,

    /// Inline shell or code in backticks (` `command` `).
    #[regex(r"`[^`\r\n]+`")]
    InlineCode,

    /// Shell subshell substitution (`$(...)`).
    #[regex(r"\$\([^\)]*\)")]
    Subshell,

    /// Environment variable expansion (`$VAR`, `${VAR}`).
    #[regex(r"\$[a-zA-Z_][a-zA-Z0-9_]*|\$\{[a-zA-Z_][a-zA-Z0-9_]*\}")]
    EnvVar,

    /// JSON object key with trailing colon (`"key":`).
    #[regex(r#""([^"\\]|\\.)*"\s*:"#)]
    JsonKeyWithColon,

    /// JSON structural delimiters: `{`, `}`, `[`, `]`, `,`, `:`.
    #[regex(r"[\{\}\[\],:]")]
    JsonDelimiter,
}

/// Scans `source` and pushes [`SpanKind::Locked`] spans into `arena` for all detected syntax tokens.
///
/// # Errors
/// Returns [`SanitizerError::InvalidSpanBounds`] if any token bounds are invalid UTF-8 char boundaries.
pub fn scan_syntax_locks(arena: &mut SpanArena<'_>) -> Result<usize, SanitizerError> {
    let source = arena.source();
    let mut lex = SyntaxToken::lexer(source);
    let mut locked_count = 0usize;

    while let Some(res) = lex.next() {
        if res.is_ok() {
            let span = lex.span();
            if span.start < span.end {
                arena.push(Span::new(span.start, span.end, SpanKind::Locked))?;
                locked_count += 1;
            }
        }
    }

    Ok(locked_count)
}

/// Verifies whether code fences, backticks, and bracket delimiters remain balanced (`AD-4`).
#[must_use]
pub fn is_syntax_balanced(text: &str) -> bool {
    // 1. If text looks like a JSON object or array, verify JSON syntax.
    let trimmed = text.trim();
    if ((trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']')))
        && serde_json::from_str::<serde_json::Value>(trimmed).is_err()
    {
        return false;
    }

    // 2. Verify balanced brackets ({}, [], ()) outside strings.
    let mut stack = Vec::with_capacity(64);
    let mut in_string = false;
    let mut escape = false;

    for ch in text.chars() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' | '[' | '(' => stack.push(ch),
            '}' if stack.pop() != Some('{') => return false,
            ']' if stack.pop() != Some('[') => return false,
            ')' if stack.pop() != Some('(') => return false,
            _ => {}
        }
    }

    if in_string || !stack.is_empty() {
        return false;
    }

    // 3. Verify balanced markdown code fences (```) and inline backticks without heap allocation.
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut fence_count = 0usize;
    let mut backtick_count = 0usize;
    let mut idx = 0usize;

    while idx < len {
        if idx + 2 < len && bytes[idx] == b'`' && bytes[idx + 1] == b'`' && bytes[idx + 2] == b'`' {
            fence_count += 1;
            idx += 3;
        } else if bytes[idx] == b'`' {
            backtick_count += 1;
            idx += 1;
        } else {
            idx += 1;
        }
    }

    if fence_count % 2 != 0 || backtick_count % 2 != 0 {
        return false;
    }

    true
}

/// Enforces the Fail-Open Syntax Invariant (`AD-4`):
/// Checks whether `candidate` maintains syntax integrity relative to `original`.
/// If syntax is corrupted by pruning, returns `false` signaling the materializer to fail open.
#[must_use]
pub fn validate_syntax_fail_open(original: &str, candidate: &str) -> bool {
    let orig_trimmed = original.trim();
    let is_orig_json = (orig_trimmed.starts_with('{') && orig_trimmed.ends_with('}'))
        || (orig_trimmed.starts_with('[') && orig_trimmed.ends_with(']'));

    if is_orig_json
        && serde_json::from_str::<serde_json::Value>(orig_trimmed).is_ok()
        && serde_json::from_str::<serde_json::Value>(candidate.trim()).is_err()
    {
        return false;
    }

    // Only enforce delimiter balance check if the original text was balanced!
    // Prevents false-positive fail-open triggers on conversational prose containing emoticons (e.g. `:)`) or lists (`1)`).
    if is_syntax_balanced(original) && !is_syntax_balanced(candidate) {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_json_delimiters_and_keys() {
        let src = r#"{"name": "Alice", "age": 30}"#;
        let mut arena = SpanArena::new(src);
        let count = scan_syntax_locks(&mut arena).unwrap_or(0);
        assert!(count >= 5);

        let locked_spans: Vec<_> = arena
            .spans()
            .iter()
            .filter(|s| matches!(s.kind, SpanKind::Locked))
            .collect();
        assert!(!locked_spans.is_empty());
    }

    #[test]
    fn locks_markdown_fences_and_backticks() {
        let src = "```python\nx = 1\n```\nRun `echo $SECRET` in $(pwd)";
        let mut arena = SpanArena::new(src);
        let count = scan_syntax_locks(&mut arena).unwrap_or(0);
        assert!(count >= 4);
    }

    #[test]
    fn detects_balanced_and_unbalanced_syntax() {
        assert!(is_syntax_balanced(r#"{"key": [1, 2, 3]}"#));
        assert!(is_syntax_balanced("```bash\necho `hello`\n```"));
        assert!(!is_syntax_balanced(r#"{"key": [1, 2, 3]"#)); // Unclosed brace
        assert!(!is_syntax_balanced("```bash\necho `hello\n```")); // Unclosed backtick
        assert!(!is_syntax_balanced("```bash\ncode")); // Unclosed fence
    }

    #[test]
    fn fail_open_validation_detects_json_corruption() {
        let orig = r#"{"user": "alice", "active": true}"#;
        let valid_cand = r#"{"user": "alice"}"#;
        let invalid_cand = r#"{"user": "alice", active true}"#;

        assert!(validate_syntax_fail_open(orig, valid_cand));
        assert!(!validate_syntax_fail_open(orig, invalid_cand));
    }
}
