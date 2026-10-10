"""Writes a golden bundle (docs/FORMATS.md) for one prompt, to check the engine against.

    python python/reference.py --model gpt2 --prompt "The human brain contains roughly" --out data/gpt2_ref.bin
"""
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer

import bundle
import hf_gpt2


def main():
    ap = argparse.ArgumentParser(description="Write a golden bundle for one prompt.")
    ap.add_argument("--model", required=True, help="Hugging Face id or local path, e.g. gpt2")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--gen", type=int, default=32, help="greedy tokens to generate")
    ap.add_argument("--out", required=True, help="output bundle path")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    model = hf_gpt2.load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = hf_gpt2.config_of(model)
    ids = torch.tensor([tok(args.prompt).input_ids])
    n = ids.shape[1]

    out = model(ids, output_hidden_states=True)
    hidden = torch.stack(out.hidden_states)[:, 0]  # [n_layer + 1, n, d]
    logits = out.logits[0]                         # [n, vocab]

    # Pin down the hidden layout FORMATS.md promises instead of trusting the HF version.
    assert hidden.shape == (cfg["n_layer"] + 1, n, cfg["d_model"]), hidden.shape
    tr = model.transformer
    embed = tr.wte.weight[ids[0]] + tr.wpe.weight[:n]
    assert torch.allclose(hidden[0], embed, atol=1e-6), "hidden[0] is not the embedding output"
    assert torch.allclose(model.lm_head(hidden[-1]), logits, atol=1e-4), "hidden[-1] is not after ln_f"

    gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.gen,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    gen_ids = gen[0, n:]
    assert len(gen_ids) == args.gen, f"generation stopped early at EOS after {len(gen_ids)} tokens"
    assert gen_ids[0] == logits[-1].argmax(), "first generated token is not the argmax of logits"

    wsum = np.array([[a.sum(dtype=np.float64), np.square(a, dtype=np.float64).sum()]
                     for _, a in hf_gpt2.weight_tensors(model)])

    bundle.write(args.out, {
        "ids": ids[0].numpy().astype(np.int32),
        "hidden": hidden.numpy(),
        "logits": logits.numpy(),
        "gen_ids": gen_ids.numpy().astype(np.int32),
        "wsum": wsum,
    })
    print(f"prompt {n} tokens, hidden {tuple(hidden.shape)}, logits {tuple(logits.shape)}, "
          f"{len(wsum)} weight sums -> {args.out}")
    print(f"{args.prompt}|{tok.decode(gen_ids)}")


if __name__ == "__main__":
    main()
