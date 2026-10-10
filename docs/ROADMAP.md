# Roadmap
[ ] todo  [~] in progress  [x] done
Detailed task files exist only for the current and next milestone. Later ones are written in the claude.ai chat when we get there.

M00 Setup and reference data (Mac)
[ ] T00.1 Repo skeleton and build
[ ] T00.2 Python environment and weight exporter
[ ] T00.3 Golden reference generator
[ ] T00.4 C++ weight loader and bundle reader
[ ] T00.5 Op fixtures
[ ] T00.6 CI

M01 CPU forward pass (Mac)
[ ] T01.0 Op stubs, tests and model skeleton
[ ] T01.1 embedding   [ ] T01.2 layernorm   [ ] T01.3 matmul   [ ] T01.4 gelu
[ ] T01.5 causal attention
[ ] T01.6 forward
[ ] T01.7 KV cache and greedy generation
[ ] T01.8 CLI and CPU benchmark
[ ] T01.9 ProtGPT2 on CPU

M02 GPU bring up and naive CUDA (Linux PC)
[ ] T02.1 Ubuntu dual boot, driver, CUDA toolkit, Tailscale SSH, remote Claude Code in tmux
[ ] T02.2 Warmup kernels and the CUDA execution model
[ ] T02.3 Device memory, error checking, GPU vs CPU op tests
[ ] T02.4 Naive kernel per op
[ ] T02.5 CUDA forward with KV cache passing golden tests
[ ] T02.6 Decode benchmark and bandwidth roofline

M03 Fast matrix kernels
[ ] T03.1 Bandwidth bound GEMV for decode, target 80 percent of 448 GB/s
[ ] T03.2 Tiled GEMM for prefill with shared memory and register blocking
[ ] T03.3 cuBLAS baseline and roofline plot
[ ] T03.4 Nsight Compute profiling workflow and write up

M04 Attention and fusion
[ ] T04.1 Decode attention over the KV cache
[ ] T04.2 Flash style prefill attention with online softmax
[ ] T04.3 Fused residual plus layernorm, fused bias plus GELU
[ ] T04.4 CUDA Graphs for the decode step

M05 Sampling for ProtGPT2
[ ] T05.1 GPU logits processors (temperature, repetition penalty 1.2, top k 950) and sampling
[ ] T05.2 Validation against HF processors and amino acid statistics
[ ] T05.3 FASTA output

M06 Reduced precision
[ ] T06.1 fp16 weights with fp32 accumulation
[ ] T06.2 Tensor core GEMM with WMMA for prefill
[ ] T06.3 int8 weight only quantization for decode
[ ] T06.4 Perplexity check on held out UniRef50 sequences

M07 Batched serving (headline result)
[ ] T07.1 Batched decode
[ ] T07.2 Paged KV cache
[ ] T07.3 Continuous batching with variable length proteins
[ ] T07.4 Throughput benchmark against HF generate, llama.cpp and vLLM, final report

M08 Optional, ESM2 650M scoring
[ ] T08.1 Bidirectional attention and RoPE
[ ] T08.2 Sequence scoring
[ ] T08.3 Generate then score pipeline
