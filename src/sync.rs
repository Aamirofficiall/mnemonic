//! Optional S3 cloud sync for mnemonic SQLite databases.
//! Enabled with `--features sync`. Disabled by default.
//!
//! Workflow:
//!   1. Server start → pull() downloads DB from S3 if local is missing or older
//!   2. All reads/writes → local SQLite (full speed, 0.1ms)
//!   3. Background task → push() uploads DB every sync_interval_secs
//!   4. Server shutdown → push() final upload

use anyhow::{anyhow, Result};
use aws_sdk_s3::Client;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::config::MnemonicConfig;

#[derive(Debug)]
pub enum SyncAction {
    Downloaded,
    AlreadyCurrent,
    RemoteNotFound,
}

pub struct SyncManager {
    client: Client,
    bucket: String,
    s3_key: String,
    local_path: PathBuf,
    interval: Duration,
    shutdown: Arc<AtomicBool>,
}

impl SyncManager {
    /// Create a SyncManager if sync is enabled in config.
    /// Returns None if sync is disabled or the `sync` feature is not compiled.
    pub async fn new(
        config: &MnemonicConfig,
        db_path: &Path,
        project_hash: &str,
    ) -> Result<Option<Self>> {
        if !config.sync_enabled {
            return Ok(None);
        }

        let region = aws_sdk_s3::config::Region::new(config.sync_region.clone());
        let aws_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(region)
            .load()
            .await;

        let client = Client::new(&aws_config);
        let s3_key = format!("{}/{}/memory.db", config.sync_prefix, project_hash);

        info!(
            "S3 sync enabled: s3://{}/{} ↔ {}",
            config.sync_bucket,
            s3_key,
            db_path.display()
        );

        Ok(Some(Self {
            client,
            bucket: config.sync_bucket.clone(),
            s3_key,
            local_path: db_path.to_path_buf(),
            interval: Duration::from_secs(config.sync_interval_secs),
            shutdown: Arc::new(AtomicBool::new(false)),
        }))
    }

    /// Pull DB from S3 if remote exists and local is missing or older.
    pub async fn pull(&self) -> Result<SyncAction> {
        // Check if remote object exists and get its last modified time
        let head = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(&self.s3_key)
            .send()
            .await;

        let remote_modified = match head {
            Ok(resp) => resp.last_modified().cloned(),
            Err(e) => {
                // Check if it's a NotFound error
                let service_err = e.into_service_error();
                if service_err.is_not_found() {
                    info!("S3 sync: no remote DB found, starting fresh");
                    return Ok(SyncAction::RemoteNotFound);
                }
                return Err(anyhow!("S3 head_object failed: {}", service_err));
            }
        };

        // Check if local file exists and its modification time
        let local_exists = self.local_path.exists();
        let local_modified = if local_exists {
            std::fs::metadata(&self.local_path)
                .ok()
                .and_then(|m| m.modified().ok())
        } else {
            None
        };

        // Decide whether to download
        let should_download = if !local_exists {
            info!("S3 sync: local DB missing, downloading from S3");
            true
        } else if let (Some(remote_ts), Some(local_ts)) = (&remote_modified, &local_modified) {
            // Convert AWS DateTime to SystemTime for comparison
            let remote_epoch = remote_ts.secs();
            let local_epoch = local_ts
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            if remote_epoch > local_epoch {
                info!(
                    "S3 sync: remote is newer (remote={}, local={}), downloading",
                    remote_epoch, local_epoch
                );
                true
            } else {
                false
            }
        } else {
            false
        };

        if !should_download {
            info!("S3 sync: local DB is current");
            return Ok(SyncAction::AlreadyCurrent);
        }

        // Download from S3
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&self.s3_key)
            .send()
            .await
            .map_err(|e| anyhow!("S3 get_object failed: {}", e))?;

        let body = resp
            .body
            .collect()
            .await
            .map_err(|e| anyhow!("S3 download body failed: {}", e))?;

        // Ensure parent directory exists
        if let Some(parent) = self.local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write to local path
        std::fs::write(&self.local_path, body.into_bytes())?;
        info!(
            "S3 sync: downloaded {} from s3://{}/{}",
            self.local_path.display(),
            self.bucket,
            self.s3_key
        );

        Ok(SyncAction::Downloaded)
    }

    /// Push local DB to S3.
    /// Uses SQLite backup API to get a consistent snapshot (safe even while DB is open).
    pub async fn push(&self) -> Result<()> {
        if !self.local_path.exists() {
            return Ok(()); // nothing to push
        }

        // Create a consistent snapshot using SQLite backup API
        // This is safe even while the main connection has the DB open with WAL
        let snapshot_path = self.local_path.with_extension("db.sync-snapshot");
        {
            let src = rusqlite::Connection::open_with_flags(
                &self.local_path,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?;
            let mut dst = rusqlite::Connection::open(&snapshot_path)?;
            let backup = rusqlite::backup::Backup::new(&src, &mut dst)?;
            backup.run_to_completion(100, std::time::Duration::from_millis(10), None)?;
        }

        let body = std::fs::read(&snapshot_path)?;
        let _ = std::fs::remove_file(&snapshot_path); // cleanup
        let byte_stream =
            aws_sdk_s3::primitives::ByteStream::from(body);

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&self.s3_key)
            .body(byte_stream)
            .send()
            .await
            .map_err(|e| anyhow!("S3 put_object failed: {}", e))?;

        info!(
            "S3 sync: uploaded {} → s3://{}/{}",
            self.local_path.display(),
            self.bucket,
            self.s3_key
        );

        Ok(())
    }

    /// Signal the background task to stop.
    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Spawn a background task that periodically pushes to S3.
    pub fn spawn_background(self: Arc<Self>) -> JoinHandle<()> {
        let interval = self.interval;
        tokio::spawn(async move {
            info!(
                "S3 sync: background push every {}s",
                interval.as_secs()
            );
            loop {
                tokio::time::sleep(interval).await;
                if self.shutdown.load(Ordering::SeqCst) {
                    break;
                }
                if let Err(e) = self.push().await {
                    warn!("S3 sync: background push failed: {}", e);
                }
            }
        })
    }
}
