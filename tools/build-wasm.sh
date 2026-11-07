#!/bin/sh
# Builds the CPU engine (wasm/) and copies it where the app loads it. Needs rustc with the wasm32 target, cargo, lld
# (all in `nix develop`).
set -eu
cd "$(dirname "$0")/../wasm"
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/tiktok_cpu.wasm ../live/cpu/tiktok_cpu.wasm
ls -la ../live/cpu/tiktok_cpu.wasm
