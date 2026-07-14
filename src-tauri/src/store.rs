use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Duration, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    File,
    Youtube,
}

impl SourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceType::File => "file",
            SourceType::Youtube => "youtube",
        }
    }

    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "file" => Ok(SourceType::File),
            "youtube" => Ok(SourceType::Youtube),
            other => anyhow::bail!("invalid source_type: {}", other),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    Transcribe,
    Translate,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Transcribe => "transcribe",
            TaskType::Translate => "translate",
        }
    }

    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "transcribe" => Ok(TaskType::Transcribe),
            "translate" => Ok(TaskType::Translate),
            other => anyhow::bail!("invalid task: {}", other),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Downloading,
    Extracting,
    Ready,
    Completed,
    Failed,
}

impl JobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Downloading => "downloading",
            JobStatus::Extracting => "extracting",
            JobStatus::Ready => "ready",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
        }
    }

    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "downloading" => Ok(JobStatus::Downloading),
            "extracting" => Ok(JobStatus::Extracting),
            "ready" => Ok(JobStatus::Ready),
            "completed" => Ok(JobStatus::Completed),
            "failed" => Ok(JobStatus::Failed),
            other => anyhow::bail!("invalid status: {}", other),
        }
    }
}

/// One persisted row of a transcript — and the ONLY shape `segments_json` ever
/// holds, in both `transcripts` and `jobs`.
///
/// **`speaker` did not ride along for free.** The frontend's word rows carried a
/// `speaker` label and this struct did not, so `#[derive(Deserialize)]` — which
/// ignores unknown fields by default — DISCARDED it on the way in, silently, and
/// the labels died at the database. It was dropped a second time on the frontend
/// side, in `segmentsForPersistence`. A field on an "opaque JSON blob" is only
/// opaque until something deserializes it into a struct; this struct is that
/// something.
///
/// Hence `deny_unknown_fields`. Every writer of `segments_json` is this very
/// struct's `Serialize`, so no row — legacy or current — can legitimately carry a
/// key it does not know, and the one place a stranger key CAN arrive is the IPC
/// boundary (`PersistTranscriptRequest.segments`, which deserializes into this
/// type). That is exactly where the silent drop happened. Now a field the
/// frontend starts sending and Rust has not learned fails `persist_transcript`
/// loudly, on the first run, instead of vanishing.
///
/// `speaker` is `Option` + `default`, which is a DIFFERENT mechanism and the one
/// that keeps legacy rows readable: a row written before diarization existed has
/// no `speaker` key at all and deserializes to `None`. `skip_serializing_if`
/// keeps the trip back out symmetric — an undiarized row serializes to
/// `{"start":…,"end":…,"text":…}`, byte-for-byte what it was before this field
/// existed, so nothing downstream (the webview included) sees a new `null` key.
/// Both are pinned by tests in this file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscriptionSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// Who said it — an OPAQUE label, e.g. `"SPEAKER_00"`, exactly as the
    /// frontend's `speakerLabel` produced it. Never parsed, never indexed by,
    /// never assumed dense or small: it is a string key, and the only thing that
    /// interprets it is `transcript_speakers.speaker_key`.
    ///
    /// `None` — the key ABSENT, not empty — whenever diarization did not run,
    /// was cancelled, degraded, or found no turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Job {
    pub id: String,
    pub source_type: SourceType,
    pub source_key: String,
    pub status: JobStatus,
    pub progress: f64,
    pub cache_hit: bool,
    pub error: Option<String>,
    pub filename: Option<String>,
    pub audio_url: Option<String>,
    pub audio_mime_type: Option<String>,
    pub model_id: String,
    pub task: TaskType,
    pub language: String,
    pub segments: Vec<TranscriptionSegment>,
    pub full_text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Source {
    pub id: i64,
    pub source_type: SourceType,
    pub source_key: String,
    pub canonical_url: Option<String>,
    pub filename: Option<String>,
    pub audio_path: Option<String>,
    pub audio_mime_type: Option<String>,
    pub size_bytes: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct CachedTranscript {
    pub full_text: String,
    pub segments: Vec<TranscriptionSegment>,
}

/// The suffix the four Whisper exports gained when they moved to their
/// word-timestamp builds.
const TIMESTAMPED_SUFFIX: &str = "_timestamped";

/// The OTHER id a transcript for `model_id` may be stored under.
///
/// The rename was purely a suffix: `onnx-community/whisper-tiny` became
/// `onnx-community/whisper-tiny_timestamped`, and the same for whisper-base,
/// whisper-large-v3-turbo and distil-small.en. So the alias is mechanical rather
/// than a hand-maintained table — a table would be a second copy of the model
/// list, which is precisely the duplication that killed the app once already.
///
/// The mapping is symmetric: a `_timestamped` id also finds a row written under
/// the legacy id, and a legacy id finds a row written under the `_timestamped`
/// one. An id with no plausible counterpart maps to `None`.
fn model_id_alias(model_id: &str) -> Option<String> {
    match model_id.strip_suffix(TIMESTAMPED_SUFFIX) {
        Some(legacy) if !legacy.is_empty() => Some(legacy.to_string()),
        Some(_) => None,
        None => Some(format!("{model_id}{TIMESTAMPED_SUFFIX}")),
    }
}

pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    pub fn open(database_path: &Path) -> anyhow::Result<Self> {
        let conn = Connection::open(database_path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // SQLite ignores every `FOREIGN KEY` clause in `SCHEMA_SQL` unless this is
        // on — it is OFF by default, per connection, for backwards compatibility.
        // The schema has always DECLARED foreign keys and never enforced one.
        //
        // What turning it on actually buys, stated narrowly because the rest would
        // be a claim with no test behind it: a `transcript_speakers` row can no
        // longer be written against a `transcript_id` that does not exist. A name
        // orphaned that way would be silently INHERITED by whatever transcript
        // next took that rowid — someone else's recording, wearing your speaker
        // names. `naming_a_speaker_of_a_transcript_that_does_not_exist_is_refused`
        // pins it, and it fails without this line.
        //
        // The `ON DELETE CASCADE` in the same clause is correct and costs nothing,
        // but it is NOT exercised today: nothing in this app deletes a transcript
        // or a source (`cleanup_expired_audio` only NULLs `audio_path`). It is
        // there for the delete path that does not exist yet, not for one that does.
        //
        // Safe for the existing tables for the same reason: enforcement applies to
        // statements run from here on, not to rows already stored, and no INSERT in
        // this file can violate the two pre-existing clauses.
        conn.pragma_update(None, "foreign_keys", true)?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn get_or_create_source(
        &self,
        source_type: SourceType,
        source_key: &str,
        canonical_url: Option<&str>,
        filename: Option<&str>,
        size_bytes: Option<i64>,
    ) -> anyhow::Result<Source> {
        let now = utc_now();
        let conn = self.conn.lock().unwrap();

        let existing: Option<i64> = conn
            .query_row(
                "SELECT id FROM sources WHERE source_type = ?1 AND source_key = ?2",
                params![source_type.as_str(), source_key],
                |row| row.get(0),
            )
            .optional()?;

        let id = match existing {
            None => {
                conn.execute(
                    "INSERT INTO sources (
                        source_type, source_key, canonical_url, filename,
                        audio_path, audio_mime_type, size_bytes,
                        created_at, updated_at, last_accessed_at
                    ) VALUES (?1, ?2, ?3, ?4, NULL, NULL, ?5, ?6, ?6, ?6)",
                    params![
                        source_type.as_str(),
                        source_key,
                        canonical_url,
                        filename,
                        size_bytes,
                        now,
                    ],
                )?;
                conn.last_insert_rowid()
            }
            Some(id) => {
                conn.execute(
                    "UPDATE sources
                     SET canonical_url = COALESCE(?1, canonical_url),
                         filename = COALESCE(?2, filename),
                         size_bytes = COALESCE(?3, size_bytes),
                         updated_at = ?4,
                         last_accessed_at = ?4
                     WHERE id = ?5",
                    params![canonical_url, filename, size_bytes, now, id],
                )?;
                id
            }
        };

        load_source(&conn, id)
    }

    pub fn touch_source(&self, source_id: i64) -> anyhow::Result<()> {
        let now = utc_now();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sources SET updated_at = ?1, last_accessed_at = ?1 WHERE id = ?2",
            params![now, source_id],
        )?;
        Ok(())
    }

    pub fn update_source_filename(&self, source_id: i64, filename: &str) -> anyhow::Result<()> {
        let now = utc_now();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sources SET filename = ?1, updated_at = ?2, last_accessed_at = ?2 WHERE id = ?3",
            params![filename, now, source_id],
        )?;
        Ok(())
    }

    pub fn update_source_audio(
        &self,
        source_id: i64,
        audio_path: &str,
        audio_mime_type: &str,
        filename: Option<&str>,
    ) -> anyhow::Result<()> {
        let now = utc_now();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sources
             SET filename = COALESCE(?1, filename),
                 audio_path = ?2,
                 audio_mime_type = ?3,
                 updated_at = ?4,
                 last_accessed_at = ?4
             WHERE id = ?5",
            params![filename, audio_path, audio_mime_type, now, source_id],
        )?;
        Ok(())
    }

    /// The transcript cache is keyed on `model_id`, so RENAMING a model orphans
    /// every transcript made under its old id — which is exactly what happened
    /// when the four models moved to their `_timestamped` exports
    /// (`onnx-community/whisper-tiny` -> `onnx-community/whisper-tiny_timestamped`,
    /// and likewise for whisper-base, whisper-large-v3-turbo and distil-small.en).
    /// Every transcript a user had already made became unreachable.
    ///
    /// So the lookup accepts the LEGACY id too, via `model_id_alias`. This was a
    /// deliberate trade: users keep every transcript they already made rather than
    /// losing them to the rename, and they avoid re-running the model on every
    /// source they have ever transcribed.
    ///
    /// The cost is permanent, not transitional. A legacy row holds
    /// sentence-granular segments with FABRICATED word times — character-length
    /// interpolation, not real DTW output — and there is no path in this app that
    /// can ever replace it with a `_timestamped` row for the same source: both
    /// `create_file_job` and `create_youtube_job` call `find_transcript` first and
    /// skip transcription entirely on a hit, and `create_job_from_cache` (the only
    /// thing they do on that hit) inserts a JOB row only — it never writes to
    /// `transcripts`. There is no re-transcribe, force, or cache-bypass path in the
    /// UI. So for any source that already has a legacy row, the exact
    /// `_timestamped` row that would out-rank it can never be created; the legacy
    /// row is a permanent hit, forever serving sentence-granular segments and
    /// fabricated word times for that source. The `ORDER BY (model_id = ?2) DESC`
    /// below — exact match wins over alias match — is correct in principle and
    /// costs nothing to keep, but as things stand today it has no legacy row left
    /// to lose to: nothing supersedes one once it exists.
    ///
    /// Consequence for callers: NEVER assume `job.segments` is word-granular just
    /// because a job came back from the cache. A legacy row's segments carry
    /// fabricated, character-length-interpolated word times, not measured ones.
    /// Speaker diarization in particular must fall back to segment-level speaker
    /// alignment for a legacy transcript — aligning speaker turns against
    /// fabricated word times is exactly the failure mode the word-timestamp work
    /// existed to prevent.
    pub fn find_transcript(
        &self,
        source_id: i64,
        model_id: &str,
        task: TaskType,
        language: &str,
    ) -> anyhow::Result<Option<CachedTranscript>> {
        let conn = self.conn.lock().unwrap();
        let row = lookup_transcript(&conn, source_id, model_id, task, language)?;

        Ok(row.map(|(_, full_text, segments_json)| CachedTranscript {
            full_text,
            segments: parse_segments(&segments_json),
        }))
    }

    /// The `transcripts` row a job's transcript actually lives in — the key that
    /// [`Store::set_speaker_name`] and [`Store::get_speaker_names`] hang off.
    ///
    /// The frontend only ever holds a JOB id, and a job is not a transcript: many
    /// jobs (one per run, plus one per cache hit) point at the same row, which is
    /// the point — a speaker renamed on Monday is still renamed when the same
    /// file is dropped in again on Friday and comes back as a cache hit.
    ///
    /// It resolves through the SAME `lookup_transcript` the cache reads with,
    /// alias and ordering included, and that sharing is load-bearing rather than
    /// tidy: if this resolved a different row than `find_transcript` serves, a
    /// user could rename the speakers of a legacy transcript and have the names
    /// attach to a row the app never reads again. Pinned by
    /// `speaker_names_attach_to_the_very_row_the_cache_serves`.
    ///
    /// `None` means the job has no transcript yet (still transcribing, or it
    /// failed) — not an error.
    pub fn transcript_id_for_job(&self, job_id: &str) -> anyhow::Result<Option<i64>> {
        let conn = self.conn.lock().unwrap();

        let recipe: Option<(Option<i64>, String, String, String)> = conn
            .query_row(
                "SELECT source_id, model_id, task, language FROM jobs WHERE id = ?1",
                params![job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;

        // A job whose source was deleted has `source_id = NULL` (ON DELETE SET
        // NULL), and a NULL source can own no transcript.
        let Some((Some(source_id), model_id, task, language)) = recipe else {
            return Ok(None);
        };
        let task = TaskType::parse(&task)?;

        Ok(lookup_transcript(&conn, source_id, &model_id, task, &language)?.map(|(id, _, _)| id))
    }

    /// Name a speaker: `SPEAKER_00` -> `"Alice"`.
    ///
    /// **A METADATA WRITE, and nothing else.** It touches one row of one table;
    /// it does not re-transcribe, does not re-diarize, and does not rewrite
    /// `segments_json`. The segments keep their opaque `SPEAKER_00` keys forever
    /// and the name is joined on at render time, which is what makes renaming
    /// free, reversible, and safe to do while the audio is long gone.
    ///
    /// An UPSERT on `(transcript_id, speaker_key)`: renaming the same speaker
    /// twice overwrites, it does not accumulate a second row that a
    /// `get_speaker_names` would then pick between arbitrarily.
    pub fn set_speaker_name(
        &self,
        transcript_id: i64,
        speaker_key: &str,
        display_name: &str,
    ) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO transcript_speakers (transcript_id, speaker_key, display_name)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(transcript_id, speaker_key)
             DO UPDATE SET display_name = excluded.display_name",
            params![transcript_id, speaker_key, display_name],
        )?;
        Ok(())
    }

    /// Every name the user has given this transcript's speakers, keyed by the
    /// opaque label the segments carry.
    ///
    /// A map, and NOT a list indexed by speaker number: the keys are strings
    /// straight from the segments (`SPEAKER_00`), and nothing here may assume they
    /// are dense, sorted, small, or even numeric. Speakers with no name simply
    /// have no entry — the caller falls back to the key itself.
    pub fn get_speaker_names(&self, transcript_id: i64) -> anyhow::Result<HashMap<String, String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT speaker_key, display_name FROM transcript_speakers WHERE transcript_id = ?1",
        )?;
        let rows = stmt.query_map(params![transcript_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut names = HashMap::new();
        for row in rows {
            let (key, name) = row?;
            names.insert(key, name);
        }
        Ok(names)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_pending_job(
        &self,
        source_id: i64,
        source_type: SourceType,
        source_key: &str,
        status: JobStatus,
        filename: Option<&str>,
        audio_mime_type: Option<&str>,
        model_id: &str,
        task: TaskType,
        language: &str,
    ) -> anyhow::Result<Job> {
        let conn = self.conn.lock().unwrap();
        let id = insert_job(
            &conn,
            source_id,
            source_type,
            source_key,
            status,
            false,
            filename,
            audio_mime_type,
            model_id,
            task,
            language,
            None,
            &[],
        )?;
        fetch_job(&conn, &id).map(|j| j.expect("job just inserted"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_job_from_cache(
        &self,
        source_id: i64,
        source_type: SourceType,
        source_key: &str,
        filename: Option<&str>,
        audio_mime_type: Option<&str>,
        model_id: &str,
        task: TaskType,
        language: &str,
        full_text: &str,
        segments: &[TranscriptionSegment],
    ) -> anyhow::Result<Job> {
        let conn = self.conn.lock().unwrap();
        let id = insert_job(
            &conn,
            source_id,
            source_type,
            source_key,
            JobStatus::Completed,
            true,
            filename,
            audio_mime_type,
            model_id,
            task,
            language,
            Some(full_text),
            segments,
        )?;
        fetch_job(&conn, &id).map(|j| j.expect("job just inserted"))
    }

    pub fn get_job(&self, job_id: &str) -> anyhow::Result<Option<Job>> {
        let conn = self.conn.lock().unwrap();
        fetch_job(&conn, job_id)
    }

    pub fn update_job(&self, job_id: &str, update: JobUpdate) -> anyhow::Result<Option<Job>> {
        let now = utc_now();
        let conn = self.conn.lock().unwrap();

        let exists: Option<String> = conn
            .query_row(
                "SELECT id FROM jobs WHERE id = ?1",
                params![job_id],
                |row| row.get(0),
            )
            .optional()?;
        if exists.is_none() {
            return Ok(None);
        }

        let mut sql = String::from("UPDATE jobs SET updated_at = ?1");
        let mut values: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(now)];

        if let Some(status) = update.status {
            sql.push_str(", status = ?");
            sql.push_str(&format!("{}", values.len() + 1));
            values.push(Box::new(status.as_str().to_string()));
        }
        if let Some(progress) = update.progress {
            sql.push_str(", progress = ?");
            sql.push_str(&format!("{}", values.len() + 1));
            values.push(Box::new(progress));
        }
        if let Some(error) = update.error {
            sql.push_str(", error = ?");
            sql.push_str(&format!("{}", values.len() + 1));
            values.push(Box::new(error));
        }
        if let Some(filename) = update.filename {
            sql.push_str(", filename = ?");
            sql.push_str(&format!("{}", values.len() + 1));
            values.push(Box::new(filename));
        }
        if let Some(mime) = update.audio_mime_type {
            sql.push_str(", audio_mime_type = ?");
            sql.push_str(&format!("{}", values.len() + 1));
            values.push(Box::new(mime));
        }

        sql.push_str(" WHERE id = ?");
        sql.push_str(&format!("{}", values.len() + 1));
        values.push(Box::new(job_id.to_string()));

        let params: Vec<&dyn rusqlite::ToSql> = values.iter().map(|b| b.as_ref()).collect();
        conn.execute(&sql, params.as_slice())?;

        fetch_job(&conn, job_id)
    }

    pub fn persist_transcript(
        &self,
        job_id: &str,
        model_id: &str,
        task: TaskType,
        language: &str,
        full_text: &str,
        segments: &[TranscriptionSegment],
    ) -> anyhow::Result<Option<Job>> {
        let now = utc_now();
        let conn = self.conn.lock().unwrap();

        let job = match fetch_job(&conn, job_id)? {
            Some(j) => j,
            None => return Ok(None),
        };

        if job.model_id != model_id || job.task != task || job.language != language {
            anyhow::bail!("Transcript payload does not match job recipe");
        }

        let source_id: i64 = conn.query_row(
            "SELECT source_id FROM jobs WHERE id = ?1",
            params![job_id],
            |row| row.get(0),
        )?;

        let segments_json = serde_json::to_string(segments)?;

        conn.execute(
            "INSERT INTO transcripts (source_id, model_id, task, language, full_text, segments_json, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
             ON CONFLICT(source_id, model_id, task, language)
             DO UPDATE SET full_text = excluded.full_text,
                           segments_json = excluded.segments_json,
                           updated_at = excluded.updated_at",
            params![source_id, model_id, task.as_str(), language, full_text, segments_json, now],
        )?;

        conn.execute(
            "UPDATE jobs SET status = ?1, progress = 1.0, error = NULL, full_text = ?2, segments_json = ?3, updated_at = ?4 WHERE id = ?5",
            params![JobStatus::Completed.as_str(), full_text, segments_json, now, job_id],
        )?;

        fetch_job(&conn, job_id)
    }

    /// Expire the prepared WAVs nobody has touched in `ttl_hours`.
    ///
    /// This used to be scoped to `source_type = 'youtube'`, because YouTube jobs
    /// were the only ones that ever wrote a WAV. Local files now write one too,
    /// so a filter on `youtube` would let every local file's WAV pile up in the
    /// audio directory forever — the app would slowly eat the disk, one
    /// transcription at a time.
    ///
    /// Expiring a local file's WAV is safe for the same reason expiring a
    /// YouTube one is: the WAV is a derived artifact, not the source. Its `Source`
    /// row survives (with `audio_path` cleared), the TRANSCRIPT survives — so a
    /// cache hit still needs no audio at all — and a fresh job for the same file
    /// simply re-runs ffmpeg over the original, which is still sitting on the
    /// user's disk.
    pub fn cleanup_expired_audio(&self, ttl_hours: i64, _audio_dir: &Path) -> usize {
        let cutoff = Utc::now() - Duration::hours(ttl_hours);
        let cutoff_str = cutoff.to_rfc3339();
        let conn = self.conn.lock().unwrap();

        let mut removed = 0usize;
        let rows: Vec<(i64, String)> = match conn
            .prepare(
                "SELECT id, audio_path FROM sources
                 WHERE audio_path IS NOT NULL
                   AND last_accessed_at < ?1",
            )
            .and_then(|mut stmt| {
                let rows: rusqlite::Result<Vec<(i64, String)>> = stmt
                    .query_map(params![cutoff_str], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect();
                rows
            }) {
            Ok(rows) => rows,
            Err(_) => return 0,
        };

        for (id, path) in rows {
            let p = PathBuf::from(&path);
            let _ = std::fs::remove_file(&p);
            let now = utc_now();
            let _ = conn.execute(
                "UPDATE sources SET audio_path = NULL, audio_mime_type = NULL, updated_at = ?1 WHERE id = ?2",
                params![now, id],
            );
            removed += 1;
        }
        removed
    }
}

#[derive(Default, Clone)]
pub struct JobUpdate {
    pub status: Option<JobStatus>,
    pub progress: Option<f64>,
    pub error: Option<String>,
    pub filename: Option<String>,
    pub audio_mime_type: Option<String>,
}

fn load_source(conn: &Connection, id: i64) -> anyhow::Result<Source> {
    let source = conn.query_row(
        "SELECT id, source_type, source_key, canonical_url, filename, audio_path, audio_mime_type, size_bytes
         FROM sources WHERE id = ?1",
        params![id],
        |row| {
            Ok(Source {
                id: row.get(0)?,
                source_type: SourceType::parse(&row.get::<_, String>(1)?).unwrap_or(SourceType::File),
                source_key: row.get(2)?,
                canonical_url: row.get(3)?,
                filename: row.get(4)?,
                audio_path: row.get(5)?,
                audio_mime_type: row.get(6)?,
                size_bytes: row.get(7)?,
            })
        },
    )?;
    Ok(source)
}

#[allow(clippy::too_many_arguments)]
fn insert_job(
    conn: &Connection,
    source_id: i64,
    source_type: SourceType,
    source_key: &str,
    status: JobStatus,
    cache_hit: bool,
    filename: Option<&str>,
    audio_mime_type: Option<&str>,
    model_id: &str,
    task: TaskType,
    language: &str,
    full_text: Option<&str>,
    segments: &[TranscriptionSegment],
) -> anyhow::Result<String> {
    let now = utc_now();
    let id = short_job_id();
    let segments_json = serde_json::to_string(segments)?;
    let progress = match status {
        JobStatus::Ready | JobStatus::Completed => 1.0,
        _ => 0.0,
    };

    conn.execute(
        "INSERT INTO jobs (
            id, source_id, source_type, source_key, status, progress,
            cache_hit, error, filename, audio_mime_type, model_id, task,
            language, full_text, segments_json, created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?15)",
        params![
            id,
            source_id,
            source_type.as_str(),
            source_key,
            status.as_str(),
            progress,
            cache_hit as i32,
            filename,
            audio_mime_type,
            model_id,
            task.as_str(),
            language,
            full_text,
            segments_json,
            now,
        ],
    )?;
    Ok(id)
}

fn fetch_job(conn: &Connection, job_id: &str) -> anyhow::Result<Option<Job>> {
    let row: Option<Job> = conn
        .query_row(
            "SELECT
                jobs.id, jobs.source_type, jobs.source_key, jobs.status, jobs.progress,
                jobs.cache_hit, jobs.error, jobs.filename, jobs.audio_mime_type,
                jobs.model_id, jobs.task, jobs.language, jobs.full_text, jobs.segments_json,
                sources.audio_path, sources.audio_mime_type, sources.filename
             FROM jobs
             LEFT JOIN sources ON sources.id = jobs.source_id
             WHERE jobs.id = ?1",
            params![job_id],
            hydrate_job,
        )
        .optional()?;
    Ok(row)
}

fn hydrate_job(row: &Row) -> rusqlite::Result<Job> {
    let id: String = row.get(0)?;
    let source_type_str: String = row.get(1)?;
    let source_type = SourceType::parse(&source_type_str).unwrap_or(SourceType::File);
    let status_str: String = row.get(3)?;
    let status = JobStatus::parse(&status_str).unwrap_or(JobStatus::Failed);
    let task_str: String = row.get(10)?;
    let task = TaskType::parse(&task_str).unwrap_or(TaskType::Transcribe);

    let job_filename: Option<String> = row.get(7)?;
    let job_mime: Option<String> = row.get(8)?;
    let source_audio_path: Option<String> = row.get(14)?;
    let source_mime: Option<String> = row.get(15)?;
    let source_filename: Option<String> = row.get(16)?;

    let segments_json: Option<String> = row.get(13)?;
    let segments = segments_json
        .as_deref()
        .map(parse_segments)
        .unwrap_or_default();

    let audio_url = if source_audio_path.is_some() {
        Some(format!("remedy-audio://{}", id))
    } else {
        None
    };

    Ok(Job {
        id,
        source_type,
        source_key: row.get(2)?,
        status,
        progress: row.get(4)?,
        cache_hit: row.get::<_, i64>(5)? != 0,
        error: row.get(6)?,
        filename: job_filename.or(source_filename),
        audio_url,
        audio_mime_type: job_mime.or(source_mime),
        model_id: row.get(9)?,
        task,
        language: row.get(11)?,
        segments,
        full_text: row.get(12)?,
    })
}

/// The ONE query that decides which `transcripts` row answers for a recipe:
/// `(id, full_text, segments_json)`, or `None`.
///
/// Shared by `find_transcript` (which reads the transcript) and
/// `transcript_id_for_job` (which hangs speaker NAMES off it) so the two cannot
/// drift into disagreeing about which row that is — see the doc comment on
/// `transcript_id_for_job` for what disagreeing would cost.
///
/// `model_id IN (?2, ?3)` + `ORDER BY (model_id = ?2) DESC` is the legacy-alias
/// lookup described at length on `find_transcript`: an exact match outranks a
/// pre-rename one.
fn lookup_transcript(
    conn: &Connection,
    source_id: i64,
    model_id: &str,
    task: TaskType,
    language: &str,
) -> anyhow::Result<Option<(i64, String, String)>> {
    // Falls back to `model_id` itself when there is no alias, which makes the
    // `IN (?2, ?3)` an exact match — the pre-rename behaviour.
    let alias = model_id_alias(model_id).unwrap_or_else(|| model_id.to_string());

    let row = conn
        .query_row(
            "SELECT id, full_text, segments_json FROM transcripts
             WHERE source_id = ?1 AND model_id IN (?2, ?3) AND task = ?4 AND language = ?5
             ORDER BY (model_id = ?2) DESC
             LIMIT 1",
            params![source_id, model_id, alias, task.as_str(), language],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    Ok(row)
}

/// A stored `segments_json` back into rows.
///
/// A parse failure yields an EMPTY transcript, which is a bad answer — so it is
/// no longer a silent one. Nothing can produce it today: every writer of
/// `segments_json` is `TranscriptionSegment`'s own `Serialize` (see
/// `deny_unknown_fields` there), and a row missing `speaker` is not a failure but
/// the ordinary legacy case. If this ever prints, the on-disk shape and the
/// struct have parted company and the transcript is being served as nothing at
/// all.
fn parse_segments(raw: &str) -> Vec<TranscriptionSegment> {
    match serde_json::from_str(raw) {
        Ok(segments) => segments,
        Err(e) => {
            eprintln!(
                "a stored transcript's segments could not be parsed and are being \
                 served as EMPTY — the row's shape does not match TranscriptionSegment: {e}"
            );
            Vec::new()
        }
    }
}

fn utc_now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
}

fn short_job_id() -> String {
    Uuid::new_v4().to_string().chars().take(8).collect()
}

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_key TEXT NOT NULL,
    canonical_url TEXT,
    filename TEXT,
    audio_path TEXT,
    audio_mime_type TEXT,
    size_bytes INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    UNIQUE(source_type, source_key)
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    task TEXT NOT NULL,
    language TEXT NOT NULL,
    full_text TEXT NOT NULL,
    segments_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_id, model_id, task, language),
    FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_id INTEGER,
    source_type TEXT NOT NULL,
    source_key TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    filename TEXT,
    audio_mime_type TEXT,
    model_id TEXT NOT NULL,
    task TEXT NOT NULL,
    language TEXT NOT NULL,
    full_text TEXT,
    segments_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE SET NULL
);

-- What the user CALLS each speaker. The segments themselves keep the opaque
-- `SPEAKER_00` keys forever; this table is the join, so a rename is a metadata
-- write and never a re-transcribe.
--
-- `CREATE TABLE IF NOT EXISTS` is purely additive, which is the whole migration
-- story here: there is no migration framework, and an existing database picks
-- this up on its next open with every other table untouched.
CREATE TABLE IF NOT EXISTS transcript_speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL,
    speaker_key TEXT NOT NULL,
    display_name TEXT NOT NULL,
    UNIQUE(transcript_id, speaker_key),
    FOREIGN KEY(transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcripts_lookup
ON transcripts(source_id, model_id, task, language);

CREATE INDEX IF NOT EXISTS idx_transcript_speakers_lookup
ON transcript_speakers(transcript_id);

CREATE INDEX IF NOT EXISTS idx_jobs_source
ON jobs(source_id);
"#;

#[allow(dead_code)]
pub fn parse_iso(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text).ok().map(|dt| dt.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempStore {
        store: Store,
        path: PathBuf,
    }

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn temp_store() -> TempStore {
        let path = std::env::temp_dir().join(format!("remedy-store-{}.sqlite", Uuid::new_v4()));
        let store = Store::open(&path).expect("open a fresh store");
        TempStore { store, path }
    }

    /// A store with one source in it — the source every test below transcribes.
    fn store_with_source() -> (TempStore, i64) {
        let temp = temp_store();
        let source = temp
            .store
            .get_or_create_source(
                SourceType::File,
                "sha256",
                None,
                Some("lecture.mp3"),
                Some(1),
            )
            .expect("create source");
        let source_id = source.id;
        (temp, source_id)
    }

    /// Write a transcript the way the app does: a job, then `persist_transcript`
    /// against that job's own recipe. Returns the JOB id — which is the only
    /// handle the frontend ever has, and therefore the handle the speaker-name
    /// commands take.
    fn write_transcript(
        store: &Store,
        source_id: i64,
        model_id: &str,
        full_text: &str,
    ) -> String {
        write_segments(
            store,
            source_id,
            model_id,
            full_text,
            &[TranscriptionSegment {
                start: 0.0,
                end: 2.5,
                text: " Hello there.".to_string(),
                speaker: None,
            }],
        )
    }

    /// The same, for a caller that cares what the segments actually are.
    fn write_segments(
        store: &Store,
        source_id: i64,
        model_id: &str,
        full_text: &str,
        segments: &[TranscriptionSegment],
    ) -> String {
        let job = store
            .create_pending_job(
                source_id,
                SourceType::File,
                "sha256",
                JobStatus::Ready,
                Some("lecture.mp3"),
                Some("audio/mpeg"),
                model_id,
                TaskType::Transcribe,
                "english",
            )
            .expect("create job");

        store
            .persist_transcript(
                &job.id,
                model_id,
                TaskType::Transcribe,
                "english",
                full_text,
                segments,
            )
            .expect("persist transcript");

        job.id
    }

    /// The rename to the `_timestamped` exports changed the cache key, which made
    /// every transcript a user had already made unreachable. The lookup now
    /// accepts the legacy id, so they keep them.
    #[test]
    fn a_transcript_written_under_the_legacy_model_id_is_found_under_the_new_one() {
        let (temp, source_id) = store_with_source();

        write_transcript(
            &temp.store,
            source_id,
            "onnx-community/whisper-base",
            "legacy transcript",
        );

        let found = temp
            .store
            .find_transcript(
                source_id,
                "onnx-community/whisper-base_timestamped",
                TaskType::Transcribe,
                "english",
            )
            .expect("query")
            .expect("the legacy row must be reachable under the renamed model id");

        assert_eq!(found.full_text, "legacy transcript");
        assert_eq!(found.segments.len(), 1);
    }

    /// The exact match wins, so re-transcribing under the new id gives the user
    /// the real word times rather than the coarse legacy row forever.
    #[test]
    fn an_exact_match_beats_the_legacy_alias() {
        let (temp, source_id) = store_with_source();

        write_transcript(
            &temp.store,
            source_id,
            "onnx-community/whisper-base",
            "legacy transcript",
        );
        write_transcript(
            &temp.store,
            source_id,
            "onnx-community/whisper-base_timestamped",
            "word-timed transcript",
        );

        let found = temp
            .store
            .find_transcript(
                source_id,
                "onnx-community/whisper-base_timestamped",
                TaskType::Transcribe,
                "english",
            )
            .expect("query")
            .expect("row");

        assert_eq!(found.full_text, "word-timed transcript");
    }

    /// A miss is still a miss: the alias must not turn an unrelated model, task or
    /// language into a false cache hit.
    #[test]
    fn the_alias_does_not_widen_the_key_beyond_the_rename() {
        let (temp, source_id) = store_with_source();

        write_transcript(
            &temp.store,
            source_id,
            "onnx-community/whisper-base",
            "legacy transcript",
        );

        const TINY: &str = "onnx-community/whisper-tiny_timestamped";
        const BASE: &str = "onnx-community/whisper-base_timestamped";

        for (model_id, task, language) in [
            (TINY, TaskType::Transcribe, "english"),
            (BASE, TaskType::Translate, "english"),
            (BASE, TaskType::Transcribe, "spanish"),
        ] {
            assert!(
                temp.store
                    .find_transcript(source_id, model_id, task, language)
                    .expect("query")
                    .is_none(),
                "{model_id}/{task:?}/{language} must not hit the whisper-base legacy row"
            );
        }
    }

    #[test]
    fn model_id_aliases_map_both_ways() {
        assert_eq!(
            model_id_alias("onnx-community/distil-small.en").as_deref(),
            Some("onnx-community/distil-small.en_timestamped")
        );
        assert_eq!(
            model_id_alias("onnx-community/whisper-large-v3-turbo_timestamped").as_deref(),
            Some("onnx-community/whisper-large-v3-turbo")
        );
        assert_eq!(model_id_alias("_timestamped"), None);
    }

    /// The lifecycle a LOCAL FILE now goes through, which it did not before.
    ///
    /// A file job used to be created straight into `Ready` with no audio anywhere
    /// — the webview decoded the file itself and Rust never saw it. It now walks
    /// the same `Extracting` -> ffmpeg -> `Ready` path a YouTube job walks, and
    /// lands with a prepared WAV recorded on its `Source`. Rust-side diarization
    /// depends on that being true for EVERY job, so it is asserted here rather
    /// than assumed.
    #[test]
    fn a_local_file_job_reaches_ready_with_prepared_audio_on_disk() {
        let (temp, source_id) = store_with_source();

        let job = temp
            .store
            .create_pending_job(
                source_id,
                SourceType::File,
                "sha256",
                JobStatus::Extracting,
                Some("lecture.mp3"),
                None,
                "onnx-community/whisper-base_timestamped",
                TaskType::Transcribe,
                "english",
            )
            .expect("create job");

        // Extraction is in flight: no audio yet, and the frontend must not be told
        // to go fetch one.
        assert!(matches!(job.status, JobStatus::Extracting));
        assert!(job.audio_url.is_none());

        // What `prepare_file_audio` does once ffmpeg has produced the WAV.
        temp.store
            .update_source_audio(source_id, "/audio/file-sha256.wav", "audio/wav", None)
            .expect("record the prepared wav");
        temp.store
            .update_job(
                &job.id,
                JobUpdate {
                    status: Some(JobStatus::Ready),
                    progress: Some(1.0),
                    audio_mime_type: Some("audio/wav".to_string()),
                    ..Default::default()
                },
            )
            .expect("update job");

        let ready = temp
            .store
            .get_job(&job.id)
            .expect("query")
            .expect("the job still exists");

        assert!(matches!(ready.status, JobStatus::Ready));
        assert_eq!(ready.progress, 1.0);
        assert_eq!(ready.audio_mime_type.as_deref(), Some("audio/wav"));
        assert!(
            ready.audio_url.is_some(),
            "a Ready file job must advertise prepared audio — the frontend fetches \
             the WAV back exactly as it does for YouTube"
        );
    }

    /// A local file's prepared WAV expires like any other.
    ///
    /// `cleanup_expired_audio` used to filter on `source_type = 'youtube'`, which
    /// was correct only while YouTube jobs were the only ones writing a WAV. Now
    /// that local files write one too, that filter would let their WAVs pile up in
    /// the audio directory forever.
    #[test]
    fn an_expired_local_file_wav_is_cleaned_up_like_a_youtube_one() {
        let temp = temp_store();
        let audio_dir = std::env::temp_dir().join(format!("remedy-audio-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&audio_dir).expect("audio dir");

        let mut written = Vec::new();
        for (source_type, key) in [
            (SourceType::File, "sha256-of-a-local-file"),
            (SourceType::Youtube, "dQw4w9WgXcQ"),
        ] {
            let source = temp
                .store
                .get_or_create_source(source_type, key, None, Some("clip"), Some(1))
                .expect("create source");

            let wav = audio_dir.join(format!("{}-{}.wav", source_type.as_str(), key));
            std::fs::write(&wav, b"RIFF").expect("write a wav");

            temp.store
                .update_source_audio(source.id, wav.to_string_lossy().as_ref(), "audio/wav", None)
                .expect("record the wav");
            written.push(wav);
        }

        // A negative TTL puts the cutoff in the future, so everything just written
        // is already expired.
        let removed = temp.store.cleanup_expired_audio(-1, &audio_dir);

        assert_eq!(removed, 2, "both source types' WAVs must expire");
        for wav in &written {
            assert!(
                !wav.exists(),
                "{} survived cleanup — the audio directory grows without bound",
                wav.display()
            );
        }

        let _ = std::fs::remove_dir_all(&audio_dir);
    }

    // -- speakers ------------------------------------------------------------

    /// The EXACT rows the frontend writes for a diarized transcript, as
    /// `segmentsForPersistence` produces them.
    ///
    /// Not retyped here — `include_str!`'d from the frontend, and pinned on that
    /// side by "writes the speaker onto the rows — and writes the shape Rust
    /// reads" in `workerTranscript.test.ts`. That is deliberate, and it is the
    /// alarm that was missing when `speaker` was silently dropped by BOTH ends of
    /// this wire: a field added to the frontend's row and not to
    /// `TranscriptionSegment` now fails HERE, loudly, because `deny_unknown_fields`
    /// refuses to parse it — instead of vanishing into a blob nobody diffs.
    ///
    /// The precedent is `commands.rs`'s `include_str!` of the frontend's model
    /// config, and the reason is the same one: a second, hand-maintained copy of a
    /// shape the other language owns is a thing that drifts, and this shape has
    /// already drifted once.
    const FRONTEND_PERSISTED_ROWS: &str =
        include_str!("../../frontend/src/lib/persistedSegments.fixture.json");

    fn diarized_rows() -> Vec<TranscriptionSegment> {
        serde_json::from_str(FRONTEND_PERSISTED_ROWS)
            .expect("the frontend's persisted row shape must deserialize into TranscriptionSegment")
    }

    /// **The whole point of the task.**
    ///
    /// Diarize a transcript, persist it, reload it: the speaker labels must still
    /// be there. They were not, and nothing failed — `segmentsForPersistence` built
    /// each row by hand as `{start, end, text}` and `TranscriptionSegment` had no
    /// `speaker` field, so the label was dropped twice over, silently, and a
    /// reopened transcript came back unlabelled.
    ///
    /// This goes through the real store: `persist_transcript` serializes to
    /// `segments_json`, `find_transcript` reads that column back and deserializes
    /// it. Handing pre-labelled segments straight to a formatter would prove
    /// nothing about any of it.
    #[test]
    fn a_diarized_transcript_survives_the_round_trip_through_the_database() {
        let (temp, source_id) = store_with_source();
        let rows = diarized_rows();

        // The premise: these rows really are diarized, and by more than one person.
        assert_eq!(rows[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(rows[8].speaker.as_deref(), Some("SPEAKER_01"));

        write_segments(
            &temp.store,
            source_id,
            "onnx-community/whisper-base_timestamped",
            "Hello there and welcome. Word-level timing is real now.",
            &rows,
        );

        let found = temp
            .store
            .find_transcript(
                source_id,
                "onnx-community/whisper-base_timestamped",
                TaskType::Transcribe,
                "english",
            )
            .expect("query")
            .expect("the transcript just written");

        assert_eq!(
            found.segments, rows,
            "the reloaded segments must be the persisted ones, speakers and all"
        );
    }

    /// A job carries its segments too (`jobs.segments_json`, the column the
    /// frontend actually renders from after a cache hit), and it is a SECOND
    /// serialization of the same struct through a different column. The label has
    /// to survive that one as well, or a cache hit renders unlabelled.
    #[test]
    fn a_cache_hit_hands_the_frontend_back_its_speakers() {
        let (temp, source_id) = store_with_source();
        let rows = diarized_rows();

        write_segments(
            &temp.store,
            source_id,
            "onnx-community/whisper-base_timestamped",
            "full text",
            &rows,
        );

        let cached = temp
            .store
            .find_transcript(
                source_id,
                "onnx-community/whisper-base_timestamped",
                TaskType::Transcribe,
                "english",
            )
            .expect("query")
            .expect("row");

        // Exactly what `create_file_job` does on a cache hit.
        let job = temp
            .store
            .create_job_from_cache(
                source_id,
                SourceType::File,
                "sha256",
                Some("lecture.mp3"),
                Some("audio/wav"),
                "onnx-community/whisper-base_timestamped",
                TaskType::Transcribe,
                "english",
                &cached.full_text,
                &cached.segments,
            )
            .expect("create job from cache");

        assert_eq!(
            job.segments, rows,
            "the job the frontend renders must carry the speakers too"
        );
    }

    /// **The legacy row must not break.** A transcript written before speakers
    /// existed has NO `speaker` key at all, and those rows persist indefinitely
    /// (see `find_transcript`: nothing supersedes one once it exists). Missing is
    /// `None`, not a parse error — `#[serde(default)]`, which is a different
    /// mechanism from `deny_unknown_fields` and the reason both can coexist.
    #[test]
    fn a_row_written_before_speakers_existed_still_reads() {
        let legacy = r#"[{"start":0.0,"end":2.5,"text":" Hello there."}]"#;

        let segments = parse_segments(legacy);
        assert_eq!(
            segments,
            vec![TranscriptionSegment {
                start: 0.0,
                end: 2.5,
                text: " Hello there.".to_string(),
                speaker: None,
            }],
            "a pre-diarization row must still parse, with no speaker"
        );
    }

    /// And an undiarized transcript must still be written EXACTLY as it always
    /// was: `speaker` absent, not `null`. The webview reads these rows, and
    /// `"speaker" in row` is the difference between "nobody said this" and a
    /// `null` the formatter would have to defend against.
    #[test]
    fn an_undiarized_segment_serializes_without_a_speaker_key_at_all() {
        let json = serde_json::to_string(&TranscriptionSegment {
            start: 0.0,
            end: 2.5,
            text: " Hello there.".to_string(),
            speaker: None,
        })
        .expect("serialize");

        assert_eq!(json, r#"{"start":0.0,"end":2.5,"text":" Hello there."}"#);
    }

    /// **A field this struct does not know about now FAILS instead of vanishing.**
    ///
    /// This is the guard that did not exist. `speaker` was sent from the frontend
    /// and thrown away here without a whisper, because serde ignores unknown fields
    /// by default. `deny_unknown_fields` turns the next such drop into an error at
    /// the first `persist_transcript` — in dev, on the first run — rather than a
    /// missing feature discovered by a user months later.
    #[test]
    fn a_field_rust_does_not_know_about_fails_loudly_instead_of_vanishing() {
        let with_a_stranger =
            r#"{"start":0.0,"end":2.5,"text":"Hello","speaker":"SPEAKER_00","confidence":0.91}"#;

        let parsed = serde_json::from_str::<TranscriptionSegment>(with_a_stranger);

        assert!(
            parsed.is_err(),
            "an unknown field must be REFUSED, not silently dropped — that silent \
             drop is exactly how speaker labels died"
        );
    }

    /// The brief's test: renaming twice must overwrite, not accumulate a second
    /// row that a later read would pick between arbitrarily.
    #[test]
    fn renaming_a_speaker_is_idempotent_and_survives_reread() {
        let (temp, source_id) = store_with_source();
        let job_id = write_transcript(
            &temp.store,
            source_id,
            "onnx-community/whisper-base_timestamped",
            "hello",
        );
        let transcript_id = temp
            .store
            .transcript_id_for_job(&job_id)
            .expect("query")
            .expect("the job has a transcript");

        temp.store
            .set_speaker_name(transcript_id, "SPEAKER_00", "Alice")
            .expect("name a speaker");
        temp.store
            .set_speaker_name(transcript_id, "SPEAKER_00", "Alice B.")
            .expect("rename her");

        let names = temp
            .store
            .get_speaker_names(transcript_id)
            .expect("read the names");

        assert_eq!(names.get("SPEAKER_00"), Some(&"Alice B.".to_string()));
        assert_eq!(names.len(), 1, "the rename must UPSERT, not accumulate");
    }

    /// **Renaming is a metadata update, and re-persisting the transcript is not a
    /// re-transcribe.** Diarization can land AFTER the transcript has been saved
    /// (it runs in a different process on different hardware), so the segments get
    /// written a second time with their labels — and the names the user has already
    /// given those speakers must survive that write. They do because
    /// `persist_transcript` UPSERTs on the recipe, so the transcript's ROWID — the
    /// key the names hang off — does not change.
    #[test]
    fn naming_survives_the_transcript_being_persisted_again() {
        let (temp, source_id) = store_with_source();
        const MODEL: &str = "onnx-community/whisper-base_timestamped";

        let job_id = write_transcript(&temp.store, source_id, MODEL, "hello");
        let transcript_id = temp
            .store
            .transcript_id_for_job(&job_id)
            .expect("query")
            .expect("transcript");

        temp.store
            .set_speaker_name(transcript_id, "SPEAKER_01", "Bob")
            .expect("name a speaker");

        // Diarization finished late: the same recipe is persisted again, this time
        // with speaker-labelled segments.
        let rows = diarized_rows();
        temp.store
            .persist_transcript(
                &job_id,
                MODEL,
                TaskType::Transcribe,
                "english",
                "hello",
                &rows,
            )
            .expect("re-persist with speakers");

        let after = temp
            .store
            .transcript_id_for_job(&job_id)
            .expect("query")
            .expect("transcript");

        assert_eq!(
            after, transcript_id,
            "an UPSERT on the recipe must keep the transcript's identity — a new \
             row would strand every name the user has given"
        );
        assert_eq!(
            temp.store
                .get_speaker_names(transcript_id)
                .expect("names")
                .get("SPEAKER_01"),
            Some(&"Bob".to_string()),
            "renaming is metadata: re-writing the segments must not erase it"
        );

        let found = temp
            .store
            .find_transcript(source_id, MODEL, TaskType::Transcribe, "english")
            .expect("query")
            .expect("row");
        assert_eq!(found.segments, rows, "and the new segments did land");
    }

    /// **The names must attach to the row the CACHE actually serves.**
    ///
    /// A legacy transcript is reachable under the renamed model id
    /// (`find_transcript`'s alias), and dropping the same file in again produces a
    /// NEW job that hits it. If `transcript_id_for_job` resolved that job to
    /// anything other than the row `find_transcript` returns, a user could rename
    /// the speakers of a transcript and have the names land on a row the app never
    /// reads — the rename would appear to work and then quietly not exist.
    ///
    /// Sharing `lookup_transcript` between the two is what makes this hold; this
    /// test is what would notice if someone stopped sharing it.
    #[test]
    fn speaker_names_attach_to_the_very_row_the_cache_serves() {
        let (temp, source_id) = store_with_source();
        const LEGACY: &str = "onnx-community/whisper-base";
        const RENAMED: &str = "onnx-community/whisper-base_timestamped";

        // The transcript was made before the model rename.
        let old_job = write_transcript(&temp.store, source_id, LEGACY, "legacy transcript");
        let transcript_id = temp
            .store
            .transcript_id_for_job(&old_job)
            .expect("query")
            .expect("the legacy job's transcript");

        temp.store
            .set_speaker_name(transcript_id, "SPEAKER_00", "Alice")
            .expect("name a speaker");

        // The user drops the same file in again, months later, under the renamed
        // model. `create_file_job` finds the legacy row through the alias and
        // creates a job straight from the cache.
        let cached = temp
            .store
            .find_transcript(source_id, RENAMED, TaskType::Transcribe, "english")
            .expect("query")
            .expect("the alias must still find the legacy row");
        let new_job = temp
            .store
            .create_job_from_cache(
                source_id,
                SourceType::File,
                "sha256",
                Some("lecture.mp3"),
                Some("audio/wav"),
                RENAMED,
                TaskType::Transcribe,
                "english",
                &cached.full_text,
                &cached.segments,
            )
            .expect("cache hit");

        assert_eq!(
            temp.store
                .transcript_id_for_job(&new_job.id)
                .expect("query"),
            Some(transcript_id),
            "a cache-hit job must resolve to the SAME transcript row the cache served it"
        );
        assert_eq!(
            temp.store
                .get_speaker_names(transcript_id)
                .expect("names")
                .get("SPEAKER_00"),
            Some(&"Alice".to_string()),
            "and the names the user gave it are still there, under a brand new job id"
        );
    }

    /// A job that has not been transcribed yet has no transcript to hang names off.
    /// That is `None`, not an error — the command turns it into an empty map on
    /// read and a refusal on write.
    #[test]
    fn a_job_with_no_transcript_yet_resolves_to_no_transcript() {
        let (temp, source_id) = store_with_source();

        let job = temp
            .store
            .create_pending_job(
                source_id,
                SourceType::File,
                "sha256",
                JobStatus::Extracting,
                Some("lecture.mp3"),
                None,
                "onnx-community/whisper-base_timestamped",
                TaskType::Transcribe,
                "english",
            )
            .expect("create job");

        assert_eq!(
            temp.store.transcript_id_for_job(&job.id).expect("query"),
            None
        );
        assert_eq!(
            temp.store.transcript_id_for_job("no-such-job").expect("query"),
            None
        );
    }

    /// A name may not be written against a transcript that does not exist.
    ///
    /// Without `PRAGMA foreign_keys = ON` (see `Store::open`) SQLite accepts the
    /// orphan happily, and the next transcript to be handed that rowid inherits it:
    /// someone else's recording, wearing your speaker names. This test fails
    /// without that pragma — the FK clause alone is decoration.
    #[test]
    fn naming_a_speaker_of_a_transcript_that_does_not_exist_is_refused() {
        let temp = temp_store();

        assert!(
            temp.store
                .set_speaker_name(9999, "SPEAKER_00", "Nobody")
                .is_err(),
            "an orphaned speaker name must be refused by the foreign key"
        );
    }
}
