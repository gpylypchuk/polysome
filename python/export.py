"""Exports any GPT2LMHeadModel checkpoint to weights v1 (docs/FORMATS.md).

    python python/export.py --model gpt2 --out data/gpt2_124M.bin
"""
import argparse
import os

import hf_gpt2


def main():
    ap = argparse.ArgumentParser(description="Export a GPT2LMHeadModel to weights v1.")
    ap.add_argument("--model", required=True, help="Hugging Face id or local path, e.g. gpt2")
    ap.add_argument("--out", required=True, help="output .bin path")
    args = ap.parse_args()

    model = hf_gpt2.load_model(args.model)
    cfg = hf_gpt2.config_of(model)
    print(args.model, " ".join(f"{k} {v}" for k, v in cfg.items()))

    # Write to a temporary file so a failed export never leaves a plausible looking .bin behind.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    n_tensors = n_floats = 0
    with open(tmp, "wb") as f:
        f.write(hf_gpt2.header(cfg).tobytes())
        for _, a in hf_gpt2.weight_tensors(model):
            f.write(a.tobytes())
            n_tensors += 1
            n_floats += a.size

    want = hf_gpt2.expected_floats(cfg)
    assert n_floats == want, f"wrote {n_floats} floats, formula says {want}"
    size = os.path.getsize(tmp)
    assert size == hf_gpt2.HEADER_BYTES + 4 * n_floats, f"file is {size} bytes"
    os.replace(tmp, args.out)
    print(f"wrote {n_tensors} tensors, {n_floats} floats, {size} bytes to {args.out}")


if __name__ == "__main__":
    main()
