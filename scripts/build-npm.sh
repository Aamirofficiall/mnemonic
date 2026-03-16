#!/bin/bash
set -euo pipefail

# Build mnemonic release binary for the current platform and copy to npm/binaries/
# Usage: ./scripts/build-npm.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
NPM_DIR="$ROOT/npm"
BIN_DIR="$NPM_DIR/binaries"

mkdir -p "$BIN_DIR"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin) PLATFORM="darwin" ;;
  Linux)  PLATFORM="linux" ;;
  MINGW*|MSYS*|CYGWIN*) PLATFORM="win32" ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
  arm64|aarch64) CPU="arm64" ;;
  x86_64|amd64)  CPU="x64" ;;
  *) echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

BIN_NAME="mnemonic-${PLATFORM}-${CPU}"
if [ "$PLATFORM" = "win32" ]; then
  BIN_NAME="${BIN_NAME}.exe"
fi

echo "Building mnemonic (release) for ${PLATFORM}-${CPU}..."
cd "$ROOT"
cargo build --release

# Find the built binary
if [ "$PLATFORM" = "win32" ]; then
  SRC="$ROOT/target/release/mnemonic.exe"
else
  SRC="$ROOT/target/release/mnemonic"
fi

if [ ! -f "$SRC" ]; then
  echo "ERROR: Binary not found at $SRC"
  exit 1
fi

cp "$SRC" "$BIN_DIR/$BIN_NAME"
chmod +x "$BIN_DIR/$BIN_NAME"

SIZE=$(du -h "$BIN_DIR/$BIN_NAME" | cut -f1)
echo ""
echo "Done! Binary: $BIN_DIR/$BIN_NAME ($SIZE)"
echo ""
echo "To test locally:"
echo "  cd npm && npm install -g ."
echo ""
echo "To publish:"
echo "  cd npm && npm publish"
