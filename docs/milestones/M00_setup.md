# M00 Setup and reference data
Goal   a repo that builds on macOS and Linux, exports any GPT2 family checkpoint to our format, and produces golden data to test against. All tasks are [CC]. Machine is the Mac.

### T00.1 [CC] Repo skeleton and build
Files  CMakeLists.txt, .gitignore, README.md, tests/check.h, tests/test_main.cpp, tools/test.sh
Spec   CMake 3.24 or newer, C++17, Release by default. Option POLYSOME_CUDA, ON when CUDAToolkit is found, with CMAKE_CUDA_ARCHITECTURES 75. tests/check.h is a dependency free harness with TEST registration, CHECK, and CHECK_CLOSE(got, want, n, atol, rtol) that reports max abs error, max rel error and the first failing index. tools/test.sh builds, runs all tests and prints one line per test plus a total. .gitignore covers build, data, .venv. README has one paragraph and an empty results table.
Done   tools/test.sh passes on macOS with one trivial test.

### T00.2 [CC] Python environment and weight exporter
Files  python/requirements.txt, python/hf_gpt2.py, python/export.py
Spec   hf_gpt2.py defines the tensor order from FORMATS.md and returns the arrays (shared by export and reference). export.py --model <hf id> --out <path> writes weights v1 for any GPT2LMHeadModel. Assert the float count against the formula.
Done   exporting gpt2 to data/gpt2_124M.bin reports 124439808 floats and the file size equals 256 + 4 times that.

### T00.3 [CC] Golden reference generator
Files  python/bundle.py, python/reference.py
Spec   reference.py --model <hf id> --prompt <text> --gen 32 --out <path> writes a golden bundle as in FORMATS.md. hidden comes from output_hidden_states, gen_ids from greedy generate (do_sample False), wsum from hf_gpt2.py arrays in float64.
Done   data/gpt2_ref.bin exists for the prompt "The human brain contains roughly" and the script prints the decoded continuation.

### T00.4 [CC] C++ weight loader and bundle reader
Files  include/polysome/config.h, include/polysome/weights.h, include/polysome/bundle.h, src/common/weights.cpp, src/common/bundle.cpp, tests/test_weights.cpp
Spec   Load weights v1 into one buffer and set per layer pointers in export order. Validate magic, version and size. The bundle reader returns named tensors with dtype and shape. The test compares per tensor sums against wsum from the golden bundle.
Done   test_weights passes, and skips with a clear message when data files are missing.

### T00.5 [CC] Op fixtures
Files  python/make_fixtures.py, tests/fixtures/*.bin
Spec   For each M01 op (embedding, layernorm, matmul with and without bias, gelu, attention, residual add) write a small seeded bundle with inputs and the expected output, computed in float64 with explicit math (no fused torch attention) and stored as float32. Shapes n 5, d 64, heads 4, out 96, so each file is a few KB and committed.
Done   fixtures exist and a test loads each one and checks shapes.

### T00.6 [CC] CI
Files  .github/workflows/ci.yml
Spec   ubuntu latest and macos latest, CPU only, build plus tools/test.sh. Tests that need data/ skip.
Done   green check on GitHub.
