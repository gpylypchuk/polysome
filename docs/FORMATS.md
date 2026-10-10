# File formats
All little endian.

## Weights v1 (.bin)
Header of 256 bytes, 64 int32 values, unused ones zero
  [0] magic 20261009   [1] version 1   [2] dtype (0 fp32, 1 fp16 later)
  [3] n_layer   [4] d_model   [5] n_head   [6] vocab   [7] n_ctx
Then tensors back to back in this order
  wte [vocab, d]   wpe [n_ctx, d]
  per layer   ln1_w [d]  ln1_b [d]  qkv_w [3d, d]  qkv_b [3d]  proj_w [d, d]  proj_b [d]
              ln2_w [d]  ln2_b [d]  fc_w [4d, d]  fc_b [4d]  fcproj_w [d, 4d]  fcproj_b [d]
  lnf_w [d]   lnf_b [d]
Total floats = vocab*d + n_ctx*d + n_layer*(12*d*d + 13*d) + 2*d
The LM head reuses wte (tied weights). HF Conv1D stores [in, out], so the exporter transposes every linear weight.

## Tensor bundle v1 (golden data and fixtures)
Header int32 magic 20261010, int32 version 1, int32 count
Per tensor   int32 name_len, name bytes (no terminator), int32 dtype (0 f32, 1 i32, 2 f64), int32 ndim, int64 dims[ndim], raw data

## Golden bundle contents
ids      i32 [n]
hidden   f32 [n_layer + 1, n, d]   index 0 is the embedding output, index i is the input of block i, and index n_layer is AFTER ln_f (HF convention, not the raw residual)
logits   f32 [n, vocab]
gen_ids  i32 [n_gen]   greedy continuation
wsum     f64 [n_tensors, 2]   sum and sum of squares per weight tensor, in weights order

## Tolerances (start values)
ops      atol 1e-5, rtol 1e-4
hidden   atol 1e-3, rtol 1e-3
logits   atol 2e-3 and identical argmax at every position
generation   identical ids
