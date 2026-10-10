# M01 CPU forward pass
Goal   reproduce PyTorch hidden states, logits and greedy generations for GPT2 124M, then ProtGPT2, on CPU. Machine is the Mac.

### T01.0 [CC] Op stubs, tests and model skeleton
Files  include/polysome/ops.h, src/cpu/ops.cpp, include/polysome/model.h, src/cpu/model.cpp, src/common/arena.cpp, tests/test_ops.cpp, tests/test_model.cpp
Spec   Signatures in ops.h
         embedding(out, ids, n, pos0, wte, wpe, d)
         layernorm(out, x, g, b, n, d)
         matmul(out, x, W, b, n, in, out_dim)      W is [out_dim, in], b may be null
         gelu(x, n)                                 in place
         residual_add(h, x, n)
         attention(out, qkv, n, d, n_head)          causal prefill
       Stubs write zeros so tests fail cleanly. test_ops checks each op against its fixture. test_model runs the golden prompt, compares hidden layer by layer and prints the first diverging layer (index n_layer is after ln_f), then logits and argmax. The arena allocates all activations once from Config.
Done   build passes and tests fail with readable messages.

### T01.1 [YOU] embedding
Spec   out[t] = wte[ids[t]] + wpe[pos0 + t]. pos0 is 0 in prefill and the current position in decode.
Done   test_ops embedding passes.
Chat   lookup as row indexing, why position is an input

### T01.2 [YOU] layernorm
Spec   per row mean and variance in two passes, eps 1e-5, then scale and shift.
Done   test_ops layernorm passes.
Chat   catastrophic cancellation, why pre LN

### T01.3 [YOU] matmul
Spec   out[t][o] = sum over i of x[t][i] * W[o][i], plus b[o] when b is not null.
Done   test_ops matmul passes, with and without bias.
Chat   row major indexing, FLOPs versus bytes, why [out, in]

### T01.4 [YOU] gelu
Spec   tanh approximation, 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))), in place.
Done   test_ops gelu passes.

### T01.5 [YOU] causal attention
Spec   each qkv row is [q | k | v], each d wide, split into n_head heads of d/n_head consecutive values. Per head and query t, score keys j <= t scaled by 1/sqrt(head_dim), subtract the max, softmax, weighted sum of v. Concatenate heads in order.
Done   test_ops attention passes.
Chat   softmax stability, O(n^2 d) cost, the seed of online softmax

### T01.6 [YOU] forward
Spec   embedding, n_layer pre LN blocks with residuals, final ln_f, LM head with wte (null bias). Buffers from the arena.
Done   test_model passes on gpt2_ref (hidden, logits, argmax).

### T01.7 [YOU] KV cache and greedy generation
Spec   forward_step(token, pos) computes one position, writes its k and v into caches [n_layer, n_ctx, d] and attends over 0..pos. generate() runs prefill, then argmax decode.
Done   generated ids equal gen_ids (32 tokens).
Chat   why the cache avoids recomputation, 2 * n_layer * d floats per token

### T01.8 [CC] CLI and CPU benchmark
Files  src/cli/main.cpp, python/run.py, bench/results/cpu.csv
Spec   The CLI reads weights and a file of token ids, generates, writes ids. run.py tokenizes, calls the CLI and detokenizes. Per op timers. CSV row with machine, commit, model, prompt length, ms per token and per op share.
Done   python/run.py prints a continuation for gpt2 and a CSV row is written.

### T01.9 [CC] ProtGPT2 on CPU
Spec   export nferruz/ProtGPT2 and its golden bundle with prompt "<|endoftext|>M". Parametrize test_model by model and run both.
Done   both models pass the model and generation tests, README results table updated.
