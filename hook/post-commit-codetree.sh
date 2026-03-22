#!/bin/bash
# mnemonic post-commit hook — auto-updates code tree on every commit
# Install: cp hook/post-commit-codetree.sh .git/hooks/post-commit && chmod +x .git/hooks/post-commit
# Or: mnemonic init (auto-installs)

# Find the project root (where .git is)
ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$ROOT" ]; then
    exit 0
fi

# Find source directories (common patterns)
for DIR in "$ROOT/src" "$ROOT/lib" "$ROOT/app" "$ROOT/Sources"; do
    if [ -d "$DIR" ]; then
        mnemonic codetree "$DIR" --update 2>/dev/null
        break
    fi
done
