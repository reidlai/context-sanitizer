//! OpenAI-compatible wire ingestion and serialization (`/v1/chat/completions`, `/v1/chat`, `/v1/completions`) (`FR-10`).

pub mod chat;
pub mod completion;

pub use chat::{sanitize_chat_completions_json, ChatCompletionRequest, ChatContent, ChatMessage};
pub use completion::{sanitize_completions_json, CompletionPrompt, CompletionRequest};
