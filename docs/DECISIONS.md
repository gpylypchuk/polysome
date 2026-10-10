# Decisions
One line per decision with its reason. Newest last.

D01 The engine supports the GPT2 architecture family. The same code runs GPT2 124M for fast iteration and ProtGPT2 738M as the target.
D02 Correctness first in fp32 on CPU, then CUDA, then lower precision.
D03 Linear weights stored [out, in] so each output row is contiguous, which makes decode GEMV reads coalesced on the GPU.
D04 Own binary formats and no third party C++ dependencies, so every byte path is understood.
D05 Tokenization stays in Python. The engine consumes and produces token ids.
D06 Separate forward implementations per backend sharing config, loader and op signatures, because fused CUDA kernels will diverge from the CPU op list.
D07 Hardware target RTX 2060 Super, Turing sm_75, 8 GB, 448 GB/s. No bf16 and no async copies on this architecture.
D08 CPU milestones on macOS. From M02 on, native Ubuntu 24.04 on the GPU PC, reached over SSH with Tailscale.
