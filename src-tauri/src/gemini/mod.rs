//! Gemini transcription engine support (credentials, chunk planning, and the
//! HTTP client today; transcript assembly lands in a later task).

pub mod chunking;
pub mod client;
pub mod credentials;
