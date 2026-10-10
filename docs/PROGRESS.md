# Progress
Read this first every session. Keep it under 40 lines. At the end of each task, set Current task to the next ID, rewrite State in at most 3 lines, and add one line to Log.

## Current task
T00.4 C++ weight loader and bundle reader, in docs/milestones/M00_setup.md

## State
CMake build and test harness in place, tools/test.sh green on macOS. CUDA path untested (no nvcc on the Mac).
.venv (Python 3.12, pinned). data/gpt2_124M.bin (export.py) and data/gpt2_ref.bin (reference.py) generated.
No engine code yet. Working on the Mac (CPU milestones M00 and M01).

## Revisit
None yet

## Log
Newest first, one line per finished task, keep the last 15
2026-10-10 T00.3 bundle.py, reference.py, gpt2_ref.bin for the brain prompt, fixed SIGBUS from misaligned mmapped weights
2026-10-10 T00.2 venv, hf_gpt2.py tensor order, export.py, gpt2 exported with 124439808 floats
2026-10-09 T00.1 CMake skeleton, tests/check.h harness, test runner, tools/test.sh, README
