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

/// The policy behind `clear()`: removing a key that was never set is a
/// SUCCESS, not an error, because the caller wanted no key stored and there
/// is none. Any other failure IS reported, since the key may still be
/// sitting in the keychain.
///
/// Split out from `clear()` so the rule is testable without a keychain --
/// `keyring::Error` is a plain enum, constructible by hand in a unit test.
fn interpret_clear(result: keyring::Result<()>) -> anyhow::Result<()> {
    match result {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(anyhow::Error::new(e).context("could not clear the API key")),
    }
}

pub fn clear() -> anyhow::Result<()> {
    interpret_clear(entry()?.delete_credential())
}

/// The policy behind `is_configured()`: only a successfully retrieved
/// password counts as "configured". A locked or unavailable keychain must
/// read as "no key" -- not surfaced as an error, and not read as configured
/// -- so it routes the user to the Add-key flow instead of into a run that
/// cannot authenticate.
///
/// Split out from `is_configured()` for the same reason as `interpret_clear`:
/// testable against a hand-built `keyring::Result` with no real keychain.
fn interpret_status(result: keyring::Result<String>) -> bool {
    result.is_ok()
}

/// Never surfaces a keychain error as "configured". A locked or unavailable
/// keychain reads as "no key", which routes the user to the Add-key flow
/// instead of into a run that cannot authenticate.
pub fn is_configured() -> bool {
    match entry() {
        Ok(e) => interpret_status(e.get_password()),
        Err(_) => false,
    }
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

    /// Constructs a boxed platform error, the payload the real backends put
    /// inside `PlatformFailure`/`NoStorageAccess` -- no keychain required.
    fn platform_error() -> Box<dyn std::error::Error + Send + Sync> {
        Box::<dyn std::error::Error + Send + Sync>::from("the keychain is locked")
    }

    #[test]
    fn clearing_a_key_that_was_never_set_is_a_success() {
        assert!(interpret_clear(Err(keyring::Error::NoEntry)).is_ok());
    }

    #[test]
    fn clearing_an_existing_key_is_a_success() {
        assert!(interpret_clear(Ok(())).is_ok());
    }

    /// Unlike a missing entry, a keychain that refuses to cooperate is a real
    /// failure: the key may still be sitting there, so the caller must not
    /// be told it is gone.
    #[test]
    fn a_locked_keychain_fails_clear_and_says_so() {
        let err = interpret_clear(Err(keyring::Error::NoStorageAccess(platform_error())))
            .expect_err("a locked keychain must not read as cleared");
        assert!(
            err.to_string().contains("could not clear"),
            "message should say what failed: {err}"
        );
    }

    #[test]
    fn a_retrieved_password_reads_as_configured() {
        assert!(interpret_status(Ok("AIzaSyABC".to_string())));
    }

    #[test]
    fn no_stored_entry_reads_as_not_configured() {
        assert!(!interpret_status(Err(keyring::Error::NoEntry)));
    }

    /// The rule this test exists to pin down: a keychain that cannot be
    /// read -- locked, unavailable, whatever the platform reason -- must
    /// read as "no key", not "configured". Reading it as configured would
    /// route a run into the Gemini API with no key to send, straight into a
    /// 401, instead of into the Add-key flow.
    #[test]
    fn a_locked_or_unavailable_keychain_reads_as_not_configured() {
        assert!(!interpret_status(Err(keyring::Error::NoStorageAccess(
            platform_error()
        ))));
        assert!(!interpret_status(Err(keyring::Error::PlatformFailure(
            platform_error()
        ))));
    }
}
