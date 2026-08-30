//! The ONLY place the Gemini API key is read or written.
//!
//! `load` is `pub(crate)` on purpose: there is no Tauri command that returns
//! the key, so once it is set it has no path back into the webview. The
//! frontend can ask WHETHER one is configured, never what it is.

use anyhow::{bail, Context};

pub const SERVICE: &str = "remedy-transcription";
pub const ACCOUNT: &str = "gemini-api-key";

/// Reject what must never reach the keychain, and normalise what should.
///
/// Separated from the keychain call so it is testable without a keychain --
/// CI has no Secret Service, and a test that needs one is a test that does not
/// run.
pub fn validate(key: &str) -> anyhow::Result<String> {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        bail!("the API key is blank");
    }
    if trimmed.contains(['\n', '\r']) {
        bail!("the API key contains a line break — paste a single line");
    }
    Ok(trimmed.to_string())
}

fn entry() -> anyhow::Result<keyring::Entry> {
    keyring::Entry::new(SERVICE, ACCOUNT).context("could not open the OS keychain")
}

pub fn save(key: &str) -> anyhow::Result<()> {
    let validated = validate(key)?;
    entry()?
        .set_password(&validated)
        .context("could not write the API key to the OS keychain")
}

/// Clearing a key that was never set is a SUCCESS, not an error: the caller
/// wanted no key stored, and there is none.
pub fn clear() -> anyhow::Result<()> {
    match entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(anyhow::Error::new(e).context("could not clear the API key")),
    }
}

/// Never surfaces a keychain error as "configured". A locked or unavailable
/// keychain reads as "no key", which routes the user to the Add-key flow
/// instead of into a run that cannot authenticate.
pub fn is_configured() -> bool {
    matches!(entry().and_then(|e| Ok(e.get_password().is_ok())), Ok(true))
}

// Not called yet: the Gemini API client (a later task) reads the key through
// this to authenticate requests. Allowed dead here rather than adding a
// premature caller, so `cargo check` stays warning-clean at this checkpoint.
#[allow(dead_code)]
pub(crate) fn load() -> anyhow::Result<String> {
    entry()?
        .get_password()
        .context("no Gemini API key is stored — add one in the app's settings")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key must never be storable blank: a blank entry is
    /// indistinguishable from "configured" to `is_configured`, so every run
    /// would sail past the no-key gate and fail at the API with a 401.
    #[test]
    fn a_blank_key_is_rejected_before_it_reaches_the_keychain() {
        let err = validate("   ").expect_err("blank must not validate");
        assert!(
            err.to_string().contains("blank"),
            "message should say why: {err}"
        );
    }

    #[test]
    fn surrounding_whitespace_is_trimmed_because_pasted_keys_carry_it() {
        assert_eq!(validate("  AIzaSyABC  ").unwrap(), "AIzaSyABC");
    }

    /// A newline in the middle is not whitespace to trim -- it is a sign the
    /// user pasted two lines, and it would produce an invalid HTTP header.
    #[test]
    fn an_embedded_newline_is_rejected() {
        assert!(validate("AIza\nSyABC").is_err());
    }

    #[test]
    fn the_keychain_entry_is_namespaced_to_this_app() {
        assert_eq!(SERVICE, "remedy-transcription");
        assert_eq!(ACCOUNT, "gemini-api-key");
    }
}
