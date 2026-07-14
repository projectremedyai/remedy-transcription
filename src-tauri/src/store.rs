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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
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
    /// So the lookup accepts the LEGACY id too, via `model_id_lookup_aliases`. The
    /// rows are not corrupt, merely coarser: they hold sentence-granular segments
    /// with no word timings, and the formatter's no-words path
    /// (`consolidateSegments` with `words` absent -> `tokensFromSegments`) renders
    /// them correctly — that is the whole reason serving them is safe. What a user
    /// gets back from an old row is the transcript they already had; what they
    /// avoid is re-running the model on every source they have ever transcribed.
    ///
    /// An exact match wins over an alias match (`ORDER BY`), so once a source is
    /// re-transcribed under the new id, the new row — with real DTW word times —
    /// is the one served.
    pub fn find_transcript(
        &self,
        source_id: i64,
        model_id: &str,
        task: TaskType,
        language: &str,
    ) -> anyhow::Result<Option<CachedTranscript>> {
        let conn = self.conn.lock().unwrap();
        // Falls back to `model_id` itself when there is no alias, which makes the
        // `IN (?2, ?3)` an exact match — the pre-rename behaviour.
        let alias = model_id_alias(model_id).unwrap_or_else(|| model_id.to_string());

        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT full_text, segments_json FROM transcripts
                 WHERE source_id = ?1 AND model_id IN (?2, ?3) AND task = ?4 AND language = ?5
                 ORDER BY (model_id = ?2) DESC
                 LIMIT 1",
                params![source_id, model_id, alias, task.as_str(), language],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        Ok(row.map(|(full_text, segments_json)| CachedTranscript {
            full_text,
            segments: parse_segments(&segments_json),
        }))
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

    pub fn cleanup_expired_audio(&self, ttl_hours: i64, _audio_dir: &Path) -> usize {
        let cutoff = Utc::now() - Duration::hours(ttl_hours);
        let cutoff_str = cutoff.to_rfc3339();
        let conn = self.conn.lock().unwrap();

        let mut removed = 0usize;
        let rows: Vec<(i64, String)> = match conn
            .prepare(
                "SELECT id, audio_path FROM sources
                 WHERE source_type = 'youtube'
                   AND audio_path IS NOT NULL
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

fn parse_segments(raw: &str) -> Vec<TranscriptionSegment> {
    serde_json::from_str(raw).unwrap_or_default()
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

CREATE INDEX IF NOT EXISTS idx_transcripts_lookup
ON transcripts(source_id, model_id, task, language);

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
    /// against that job's own recipe.
    fn write_transcript(store: &Store, source_id: i64, model_id: &str, full_text: &str) {
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
                &[TranscriptionSegment {
                    start: 0.0,
                    end: 2.5,
                    text: " Hello there.".to_string(),
                }],
            )
            .expect("persist transcript");
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
}
