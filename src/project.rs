use std::path::{Path, PathBuf};
use tracing::info;

/// Detect project root by walking up from CWD looking for .git/
pub fn detect_project_root() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let mut dir = cwd.as_path();

    loop {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// Generate a deterministic hash for a project path.
pub fn project_hash(root: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    root.to_string_lossy().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Get the SQLite database path for the current project.
/// Layout: ~/.mnemonic/{project-hash}/memory.db
pub fn memory_db_path() -> PathBuf {
    let home = dirs_or_home();
    let memories_dir = home.join(".mnemonic");

    if let Some(root) = detect_project_root() {
        let hash = project_hash(&root);
        let project_dir = memories_dir.join(&hash);

        // Write project index entry so we can list projects later
        let _ = write_index_entry(&memories_dir, &hash, &root);

        info!(
            "Project: {} → {}",
            root.display(),
            project_dir.display()
        );
        project_dir.join("memory.db")
    } else {
        // No git root — use CWD hash
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let hash = project_hash(&cwd);
        info!("No git root, using CWD hash: {}", hash);
        memories_dir.join(hash).join("memory.db")
    }
}

/// Write/update the project index so we can list all known projects.
fn write_index_entry(memories_dir: &Path, hash: &str, root: &Path) -> std::io::Result<()> {
    let index_path = memories_dir.join("index.json");
    let mut index: serde_json::Map<String, serde_json::Value> = if index_path.exists() {
        let data = std::fs::read_to_string(&index_path)?;
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        serde_json::Map::new()
    };

    index.insert(
        hash.to_string(),
        serde_json::json!({
            "path": root.to_string_lossy(),
            "last_seen": chrono::Utc::now().to_rfc3339(),
        }),
    );

    std::fs::create_dir_all(memories_dir)?;
    std::fs::write(
        &index_path,
        serde_json::to_string_pretty(&index).unwrap_or_default(),
    )?;
    Ok(())
}

fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}
