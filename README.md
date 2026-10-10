# polysome

polysome is an inference engine for the GPT2 model family, written from scratch in C++17 and CUDA with no third party C++ dependencies. Its target is ProtGPT2, a 738M parameter GPT2 that generates protein sequences, running at maximum throughput on a single RTX 2060 Super (Turing sm_75, 8 GB, 448 GB/s). Every op is tested against PyTorch fixtures, every model change is checked against golden outputs, and every speedup below is a measured benchmark row.

## Results

| Model | Backend | GPU | Precision | Batch | Tokens/s | Commit |
|-------|---------|-----|-----------|-------|----------|--------|
