# polysome

Inference engine written from scratch in C++ and CUDA for the GPT2 model family. Target is ProtGPT2 (738M params, protein sequence generation) at maximum throughput on an RTX 2060 Super (Turing sm_75, 8 GB, 448 GB/s). It is a learning project and a portfolio piece, so clarity and measured results matter as much as speed.

## Language
* Code, comments, commit messages and repo docs in English.
* Talk to the user (Suipaya) in Spanish, Rioplatense voseo.

## Session protocol (token budget matters)
1. Read docs/PROGRESS.md. It names the current task and the milestone file that holds it.
2. Read only that task's section of the milestone file. Do not open ROADMAP.md, other milestones or unrelated files unless the task needs them. The task lists the files to touch.
3. One task per session. Finish with tests green, a commit, and PROGRESS.md updated following its own instructions.
4. Keep output small. Run tests with tools/test.sh, pipe long output through tail, never print weights or large arrays.
5. For deep theory give a short answer and suggest taking it to the claude.ai chat.
6. When a milestone ends, tick ROADMAP.md and tell the user to plan the next milestone file in the claude.ai chat.

## Ownership and teaching mode
Tasks are tagged [CC] (you implement) or [YOU] (Suipaya implements).
For [YOU] tasks
* Open with a briefing in Spanish of at most 10 lines covering what the code computes, shapes, one pitfall and the test that checks it. Then stop.
* Never write the implementation unprompted. Escalate only when asked
  * "pista" gives a conceptual hint, no code
  * "pseudo" gives pseudocode
  * "solución" gives the full solution commented line by line
  * "seguimos" means write it with comments to keep momentum and add the task to Revisit in PROGRESS.md
* When he says "listo", run the tests, then review correctness first and add at most one note on performance or clarity.
* Before committing, ask him one question that checks he can explain the code.

## Engineering rules
* C++17. CUDA is optional through the CMake option POLYSOME_CUDA, enabled when the toolkit is found. No third party C++ dependencies.
* File formats and test tolerances live in docs/FORMATS.md. Linear weights are [out, in], row major.
* Correctness before speed. Every op has a unit test against Python fixtures and every model change is checked against golden data.
* Never edit a test or a tolerance to make it pass. Changing a tolerance needs a line in docs/DECISIONS.md.
* No allocations in the forward hot path. Activations come from one arena.
* Benchmarks append rows to bench/results/*.csv with machine, GPU, commit, model and settings.
* data/ is gitignored. Recreate it with python/export.py and python/reference.py.

## Commands
    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
    tools/test.sh
    source .venv/bin/activate

## Layout
include/polysome headers · src/common loader, bundle, arena, timer · src/cpu · src/cuda · src/cli · python · tests · bench · tools · docs
