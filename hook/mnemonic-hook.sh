#!/bin/bash
# mnemonic PostToolUse hook
# Auto-captures tool outcomes for project memory.
# Installed by: mnemonic init
#
# This script is called by Claude Code after every tool call.
# It reads the hook event from stdin and forwards it to the memory
# server's Unix socket for background processing.

SOCKET="$HOME/.mnemonic/memory.sock"

if [ -S "$SOCKET" ]; then
    # Forward to memory server (non-blocking)
    cat | socat - UNIX-CONNECT:"$SOCKET" 2>/dev/null &
fi
