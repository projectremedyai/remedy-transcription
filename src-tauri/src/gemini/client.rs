//! HTTP against the Gemini API. Knows nothing about jobs, chunks or progress.
//!
//! The base URL is injectable so the whole surface -- including the two-step
//! resumable upload, which is easy to get subtly wrong -- is exercised against
//! a local mock server rather than mocked out behind a trait.

use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, Context};

pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

#[derive(Debug, thiserror::Error)]
pub enum GeminiError {
    #[error("Gemini rejected this API key")]
    InvalidKey,
    #[error("Gemini is rate limiting this key; retry in {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
    #[error("Gemini returned a server error ({status})")]
    ServerError { status: u16 },
    #[error("Gemini returned {status}: {body}")]
    Rejected { status: u16, body: String },
}

impl GeminiError {
    /// Only conditions that a later identical request could survive.
    ///
    /// A 400 is our malformed request and a 401 is the user's key: retrying
    /// either just makes the same mistake again, more slowly.
    // Not called yet: Task 11's orchestration reads this to decide whether to
    // retry a chunk. Allowed dead here rather than adding a premature caller.
    #[allow(dead_code)]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            GeminiError::RateLimited { .. } | GeminiError::ServerError { .. }
        )
    }
}

pub fn classify_status(status: u16, retry_after: Option<u64>) -> Option<GeminiError> {
    match status {
        200..=299 => None,
        401 | 403 => Some(GeminiError::InvalidKey),
        429 => Some(GeminiError::RateLimited {
            retry_after_secs: retry_after.unwrap_or(5),
        }),
        500..=599 => Some(GeminiError::ServerError { status }),
        // Google returns 429 RESOURCE_EXHAUSTED for the 20 GB Files API project
        // quota as well as for rate limiting, so a quota problem arrives as
        // RateLimited. The body is what distinguishes them, which is why
        // `check_response` reads it -- see below.
        other => Some(GeminiError::Rejected {
            status: other,
            body: String::new(),
        }),
    }
}

// Not constructed outside tests yet: `upload` returns this to Task 11's
// orchestration, which is also the future caller of `upload` itself.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct UploadedFile {
    pub uri: String,
    /// The `files/xyz` resource name, needed to DELETE it afterwards.
    pub name: String,
}

pub struct GeminiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL.to_string())
    }

    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            // NO overall request timeout, deliberately. Transcribing 25 minutes
            // of audio legitimately takes minutes, and a timeout makes a retry
            // ambiguous -- the request may have landed, and re-issuing it
            // double-charges. Cancellation is the only stop. The CONNECT
            // timeout is safe: a connection that never opens ran nothing.
            http: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client builds with rustls"),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }

    /// Classify a response, reading the body on failure.
    ///
    /// `classify_status` stays pure and sync so the mapping is unit-testable;
    /// this consumes the response because Google's own message is the only
    /// thing that distinguishes, say, a Files API quota exhaustion from
    /// ordinary rate limiting -- both arrive as 429.
    ///
    /// Returns the response untouched on success, so callers can chain.
    async fn check_response(response: reqwest::Response) -> anyhow::Result<reqwest::Response> {
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());

        match classify_status(response.status().as_u16(), retry_after) {
            None => Ok(response),
            Some(GeminiError::Rejected { status, .. }) => {
                let body = response.text().await.unwrap_or_default();
                Err(GeminiError::Rejected { status, body }.into())
            }
            Some(GeminiError::RateLimited { retry_after_secs }) => {
                // Surface Google's own wording alongside ours: "quota exceeded"
                // and "too many requests" need different actions from the user.
                let body = response.text().await.unwrap_or_default();
                Err(anyhow::Error::new(GeminiError::RateLimited { retry_after_secs })
                    .context(body))
            }
            Some(e) => Err(e.into()),
        }
    }

    /// The Files API's two-step resumable upload.
    ///
    /// Step one declares the size and type and gets back a one-shot URL in the
    /// `x-goog-upload-url` RESPONSE HEADER -- not in the body, which is empty.
    /// Step two sends the bytes to that URL.
    // Not called yet: Task 11's orchestration uploads each chunk before
    // handing its URI to `interact`. Allowed dead here rather than adding a
    // premature caller; only the tests call it until then.
    #[allow(dead_code)]
    pub async fn upload(
        &self,
        path: &Path,
        mime: &str,
        display_name: &str,
    ) -> anyhow::Result<UploadedFile> {
        let bytes = tokio::fs::read(path)
            .await
            .with_context(|| format!("could not read {}", path.display()))?;

        let start = self
            .http
            .post(format!("{}/upload/v1beta/files", self.base_url))
            .header("x-goog-api-key", &self.api_key)
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", bytes.len().to_string())
            .header("X-Goog-Upload-Header-Content-Type", mime)
            .header("Content-Type", "application/json")
            .body(serde_json::json!({ "file": { "display_name": display_name } }).to_string())
            .send()
            .await
            .context("could not start the upload")?;

        let start = Self::check_response(start).await?;

        let upload_url = start
            .headers()
            .get("x-goog-upload-url")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| anyhow!("the upload start response carried no upload URL"))?
            .to_string();

        let finalize = self
            .http
            .post(upload_url)
            .header("Content-Length", bytes.len().to_string())
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(bytes)
            .send()
            .await
            .context("could not finalize the upload")?;

        let finalize = Self::check_response(finalize).await?;

        let body: serde_json::Value = finalize.json().await.context("malformed upload response")?;
        Ok(UploadedFile {
            uri: body["file"]["uri"]
                .as_str()
                .ok_or_else(|| anyhow!("upload response had no file.uri"))?
                .to_string(),
            name: body["file"]["name"]
                .as_str()
                .ok_or_else(|| anyhow!("upload response had no file.name"))?
                .to_string(),
        })
    }

    // Not called yet: Task 10 parses what this returns, and Task 11 is the
    // orchestration that will actually call it with a real request body.
    #[allow(dead_code)]
    pub async fn interact(&self, body: serde_json::Value) -> anyhow::Result<serde_json::Value> {
        let response = self
            .http
            .post(format!("{}/v1beta/interactions", self.base_url))
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .context("could not reach the Gemini interactions endpoint")?;

        Self::check_response(response)
            .await?
            .json()
            .await
            .context("malformed interactions response")
    }

    /// Best effort by contract. Cleanup runs on the failure and cancel paths,
    /// where turning one error into two helps nobody -- and the file expires in
    /// 48 hours regardless.
    // Not called yet: Task 11's orchestration calls this on the failure and
    // cancel paths described above.
    #[allow(dead_code)]
    pub async fn delete_file(&self, name: &str) -> anyhow::Result<()> {
        let _ = self
            .http
            .delete(format!("{}/v1beta/{}", self.base_url, name.trim_start_matches('/')))
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await;
        Ok(())
    }

    /// One cheap call to prove a key works, spent at paste time.
    pub async fn validate_key(&self) -> anyhow::Result<()> {
        let response = self
            .http
            .get(format!("{}/v1beta/models", self.base_url))
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await
            .context("could not reach Google to check the key")?;
        Self::check_response(response).await.map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;

    #[test]
    fn a_success_status_classifies_as_no_error() {
        assert!(classify_status(200, None).is_none());
    }

    /// 401 and 403 are the user's problem and retrying cannot fix them --
    /// retrying a rejected key just spends three round trips saying so.
    #[test]
    fn an_auth_failure_is_not_retryable() {
        for status in [401, 403] {
            let err = classify_status(status, None).expect("should be an error");
            assert!(matches!(err, GeminiError::InvalidKey));
            assert!(!err.is_retryable());
        }
    }

    #[test]
    fn rate_limiting_is_retryable_and_carries_the_retry_after() {
        let err = classify_status(429, Some(17)).expect("should be an error");
        assert!(err.is_retryable());
        assert!(matches!(err, GeminiError::RateLimited { retry_after_secs: 17 }));
    }

    #[test]
    fn server_errors_are_retryable() {
        assert!(classify_status(503, None).unwrap().is_retryable());
        assert!(classify_status(500, None).unwrap().is_retryable());
    }

    /// A 400 is a malformed request -- our bug, not a transient condition.
    /// Retrying it three times makes the same mistake three times.
    #[test]
    fn a_bad_request_is_not_retryable() {
        assert!(!classify_status(400, None).unwrap().is_retryable());
    }

    #[tokio::test]
    async fn upload_follows_the_two_step_resumable_protocol() {
        let server = MockServer::start_async().await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunk.flac");
        std::fs::write(&path, b"FLACfake").unwrap();

        let start = server.mock(|when, then| {
            when.method(POST)
                .path("/upload/v1beta/files")
                .header("x-goog-api-key", "KEY")
                .header("X-Goog-Upload-Protocol", "resumable")
                .header("X-Goog-Upload-Command", "start")
                .header("X-Goog-Upload-Header-Content-Length", "8")
                .header("X-Goog-Upload-Header-Content-Type", "audio/flac");
            then.status(200)
                .header("x-goog-upload-url", format!("{}/resume/abc", server.base_url()));
        });

        let finalize = server.mock(|when, then| {
            when.method(POST)
                .path("/resume/abc")
                .header("X-Goog-Upload-Offset", "0")
                .header("X-Goog-Upload-Command", "upload, finalize")
                .body("FLACfake");
            then.status(200).json_body(serde_json::json!({
                "file": { "uri": "https://x/files/xyz", "name": "files/xyz" }
            }));
        });

        let client = GeminiClient::with_base_url("KEY".into(), server.base_url());
        let uploaded = client.upload(&path, "audio/flac", "chunk").await.unwrap();

        start.assert();
        finalize.assert();
        assert_eq!(uploaded.uri, "https://x/files/xyz");
        assert_eq!(uploaded.name, "files/xyz");
    }

    #[tokio::test]
    async fn a_start_response_without_an_upload_url_is_an_error_not_a_panic() {
        let server = MockServer::start_async().await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunk.flac");
        std::fs::write(&path, b"x").unwrap();

        server.mock(|when, then| {
            when.method(POST).path("/upload/v1beta/files");
            then.status(200); // no x-goog-upload-url header
        });

        let client = GeminiClient::with_base_url("KEY".into(), server.base_url());
        let err = client.upload(&path, "audio/flac", "c").await.unwrap_err();
        assert!(err.to_string().contains("upload URL"), "got: {err}");
    }

    #[tokio::test]
    async fn interact_posts_to_the_interactions_endpoint_with_the_key() {
        let server = MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1beta/interactions")
                .header("x-goog-api-key", "KEY");
            then.status(200).json_body(serde_json::json!({"output_text": "hi"}));
        });

        let client = GeminiClient::with_base_url("KEY".into(), server.base_url());
        let body = client.interact(serde_json::json!({"model": "m"})).await.unwrap();

        mock.assert();
        assert_eq!(body["output_text"], "hi");
    }

    #[tokio::test]
    async fn a_rejected_key_surfaces_as_invalid_key() {
        let server = MockServer::start_async().await;
        server.mock(|when, then| {
            when.method(POST).path("/v1beta/interactions");
            then.status(401).body("nope");
        });

        let client = GeminiClient::with_base_url("KEY".into(), server.base_url());
        let err = client.interact(serde_json::json!({})).await.unwrap_err();
        assert!(err.to_string().contains("rejected"), "got: {err}");
    }

    /// Deleting an already-gone file must SUCCEED: cleanup runs on the failure
    /// and cancel paths too, and a cleanup that can fail turns one error into
    /// two.
    #[tokio::test]
    async fn deleting_a_missing_file_is_not_an_error() {
        let server = MockServer::start_async().await;
        server.mock(|when, then| {
            when.method(DELETE).path("/v1beta/files/xyz");
            then.status(404);
        });

        let client = GeminiClient::with_base_url("KEY".into(), server.base_url());
        assert!(client.delete_file("files/xyz").await.is_ok());
    }

    #[tokio::test]
    async fn validate_key_calls_the_models_endpoint() {
        let server = MockServer::start_async().await;
        let mock = server.mock(|when, then| {
            when.method(GET).path("/v1beta/models").header("x-goog-api-key", "KEY");
            then.status(200).json_body(serde_json::json!({"models": []}));
        });

        let client = GeminiClient::with_base_url("KEY".into(), server.base_url());
        client.validate_key().await.unwrap();
        mock.assert();
    }
}
