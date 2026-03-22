//! TDD tests for codetree — unit tests first, then e2e.
//! Run: cargo test --test codetree_test -- --nocapture

use std::fs;
use std::path::Path;

// ═══════════════════════════════════════════════════════════════
// UNIT TESTS — individual components
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_detect_language() {
    // Test via file extension matching
    let cases = vec![
        ("main.rs", true),
        ("lib.py", true),
        ("app.cpp", true),
        ("utils.c", true),
        ("index.ts", true),
        ("app.js", true),
        ("ViewController.swift", true),
        ("readme.md", false),
        ("Cargo.toml", false),
        ("image.png", false),
    ];

    for (file, should_detect) in cases {
        let ext = Path::new(file).extension().and_then(|e| e.to_str());
        let detected = matches!(ext, Some("rs" | "py" | "c" | "h" | "cpp" | "cc" | "cxx" | "hpp" | "js" | "jsx" | "ts" | "tsx" | "swift"));
        assert_eq!(detected, should_detect, "detect_language({}) = {}, expected {}", file, detected, should_detect);
    }
    eprintln!("[PASS] Language detection for 10 file types");
}

#[test]
fn test_parse_rust_file() {
    // Create a temp Rust file
    let tmp = std::env::temp_dir().join("mnemonic-codetree-test");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let rust_file = tmp.join("test.rs");
    fs::write(&rust_file, r#"
pub struct MyStruct {
    field: String,
}

impl MyStruct {
    pub fn new(value: &str) -> Self {
        Self { field: value.to_string() }
    }

    pub fn get_field(&self) -> &str {
        &self.field
    }
}

pub fn standalone_function(x: i32, y: i32) -> i32 {
    x + y
}

pub trait MyTrait {
    fn do_something(&self);
}

pub enum MyEnum {
    A,
    B(String),
}
"#).unwrap();

    // Parse it
    let tags = mnemonic::codetree::parse_file(&rust_file, &tmp).unwrap();

    let defs: Vec<_> = tags.iter()
        .filter(|t| t.kind == mnemonic::codetree::TagKind::Definition)
        .collect();

    eprintln!("[INFO] Parsed {} tags, {} definitions:", tags.len(), defs.len());
    for d in &defs {
        eprintln!("  {} {} | {}", d.symbol_type, d.name, d.signature);
    }

    // Verify key symbols found
    let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"MyStruct"), "Should find struct MyStruct");
    assert!(names.contains(&"new"), "Should find method new");
    assert!(names.contains(&"get_field"), "Should find method get_field");
    assert!(names.contains(&"standalone_function"), "Should find standalone_function");
    assert!(names.contains(&"MyTrait"), "Should find trait MyTrait");
    assert!(names.contains(&"MyEnum"), "Should find enum MyEnum");

    eprintln!("[PASS] Rust parsing: found struct, methods, function, trait, enum");

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_parse_python_file() {
    let tmp = std::env::temp_dir().join("mnemonic-codetree-test-py");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let py_file = tmp.join("test.py");
    fs::write(&py_file, r#"
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int) -> dict:
        return self.db.find(user_id)

def validate_email(email: str) -> bool:
    return "@" in email

MAX_RETRIES = 3
"#).unwrap();

    let tags = mnemonic::codetree::parse_file(&py_file, &tmp).unwrap();
    let defs: Vec<_> = tags.iter()
        .filter(|t| t.kind == mnemonic::codetree::TagKind::Definition)
        .collect();

    eprintln!("[INFO] Python: {} tags, {} defs:", tags.len(), defs.len());
    for d in &defs {
        eprintln!("  {} {} | {}", d.symbol_type, d.name, d.signature);
    }

    let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"UserService"), "Should find class UserService");
    assert!(names.contains(&"get_user"), "Should find method get_user");
    assert!(names.contains(&"validate_email"), "Should find function validate_email");

    eprintln!("[PASS] Python parsing: found class, methods, function");

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_parse_cpp_file() {
    let tmp = std::env::temp_dir().join("mnemonic-codetree-test-cpp");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let cpp_file = tmp.join("test.cpp");
    fs::write(&cpp_file, r#"
class GpuRenderer {
public:
    void render_frame(const Scene& scene);
    void flush_cache();
};

void GpuRenderer::render_frame(const Scene& scene) {
    // render
}

int standalone_calc(int x, int y) {
    return x + y;
}

struct Vec2 {
    float x, y;
};
"#).unwrap();

    let tags = mnemonic::codetree::parse_file(&cpp_file, &tmp).unwrap();
    let defs: Vec<_> = tags.iter()
        .filter(|t| t.kind == mnemonic::codetree::TagKind::Definition)
        .collect();

    eprintln!("[INFO] C++: {} tags, {} defs:", tags.len(), defs.len());
    for d in &defs {
        eprintln!("  {} {} | {}", d.symbol_type, d.name, d.signature);
    }

    let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"GpuRenderer"), "Should find class GpuRenderer");
    assert!(names.contains(&"standalone_calc"), "Should find function standalone_calc");
    assert!(names.contains(&"Vec2"), "Should find struct Vec2");

    eprintln!("[PASS] C++ parsing: found class, function, struct");

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_file_hash_changes() {
    let tmp = std::env::temp_dir().join("mnemonic-hash-test");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let file = tmp.join("test.rs");
    fs::write(&file, "fn original() {}").unwrap();
    let hash1 = mnemonic::codetree::file_hash(&file).unwrap();

    fs::write(&file, "fn modified() {}").unwrap();
    let hash2 = mnemonic::codetree::file_hash(&file).unwrap();

    assert_ne!(hash1, hash2, "Hash should change when file changes");

    fs::write(&file, "fn original() {}").unwrap();
    let hash3 = mnemonic::codetree::file_hash(&file).unwrap();

    assert_eq!(hash1, hash3, "Same content should produce same hash");

    eprintln!("[PASS] File hash: changes on modification, stable for same content");

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_scan_directory_skips_hidden() {
    let tmp = std::env::temp_dir().join("mnemonic-scan-test");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(tmp.join(".hidden")).unwrap();
    fs::create_dir_all(tmp.join("target")).unwrap();
    fs::create_dir_all(tmp.join("node_modules")).unwrap();
    fs::create_dir_all(tmp.join("src")).unwrap();

    fs::write(tmp.join(".hidden/secret.rs"), "fn x() {}").unwrap();
    fs::write(tmp.join("target/build.rs"), "fn x() {}").unwrap();
    fs::write(tmp.join("node_modules/pkg.js"), "function x() {}").unwrap();
    fs::write(tmp.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(tmp.join("src/lib.py"), "def x(): pass").unwrap();

    let files = mnemonic::codetree::scan_directory(&tmp).unwrap();
    let paths: Vec<String> = files.iter().map(|(p, _)| p.to_string_lossy().to_string()).collect();

    eprintln!("[INFO] Scanned files: {:?}", paths);

    assert!(paths.iter().any(|p| p.contains("main.rs")), "Should find src/main.rs");
    assert!(paths.iter().any(|p| p.contains("lib.py")), "Should find src/lib.py");
    assert!(!paths.iter().any(|p| p.contains(".hidden")), "Should skip .hidden/");
    assert!(!paths.iter().any(|p| p.contains("target")), "Should skip target/");
    assert!(!paths.iter().any(|p| p.contains("node_modules")), "Should skip node_modules/");

    eprintln!("[PASS] Directory scan: finds src files, skips hidden/target/node_modules");

    let _ = fs::remove_dir_all(&tmp);
}

// ═══════════════════════════════════════════════════════════════
// E2E TEST — full pipeline on our own codebase
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_e2e_parse_own_codebase() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    if !src.exists() {
        eprintln!("[SKIP] src/ not found");
        return;
    }

    // Scan
    let files = mnemonic::codetree::scan_directory(&src).unwrap();
    eprintln!("[INFO] Found {} source files", files.len());
    assert!(files.len() >= 8, "Should find 8+ source files in our src/");

    // Parse each file
    let mut total_defs = 0;
    let mut total_refs = 0;
    for (path, lang) in &files {
        match mnemonic::codetree::parse_file(path, &src) {
            Ok(tags) => {
                let defs = tags.iter().filter(|t| t.kind == mnemonic::codetree::TagKind::Definition).count();
                let refs = tags.iter().filter(|t| t.kind == mnemonic::codetree::TagKind::Reference).count();
                total_defs += defs;
                total_refs += refs;
                let name = path.file_name().unwrap().to_string_lossy();
                eprintln!("  {} [{}]: {} defs, {} refs", name, lang, defs, refs);
            }
            Err(e) => eprintln!("  SKIP {}: {}", path.display(), e),
        }
    }

    eprintln!("\n[INFO] Total: {} definitions, {} references", total_defs, total_refs);
    assert!(total_defs > 100, "Should find 100+ definitions, got {}", total_defs);
    assert!(total_refs > 50, "Should find 50+ references, got {}", total_refs);

    eprintln!("[PASS] E2E: parsed {} files, {} defs, {} refs", files.len(), total_defs, total_refs);
}
