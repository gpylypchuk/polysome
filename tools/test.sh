#!/usr/bin/env bash
# Configure if needed, build, and run every unit test.
# Usage: tools/test.sh [substring]   runs only tests whose name contains the substring.
# BUILD_DIR overrides the build directory (default: build).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
build="${BUILD_DIR:-$root/build}"
mkdir -p "$build"
log="$build/test_build.log"

# Build output is kept quiet and shown only when something fails.
if [ ! -f "$build/CMakeCache.txt" ]; then
  if ! cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release >"$log" 2>&1; then
    tail -n 40 "$log"
    exit 1
  fi
fi
if ! cmake --build "$build" -j >"$log" 2>&1; then
  tail -n 40 "$log"
  exit 1
fi

"$build/polysome_tests" "$@"
