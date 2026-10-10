"""GPT2 checkpoint access shared by export.py and reference.py.

This module owns the weights v1 layout from docs/FORMATS.md (header and tensor order),
so the exporter and the golden data can never disagree about it.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel

MAGIC = 20261009
VERSION = 1
DTYPE_FP32 = 0
HEADER_INTS = 64
HEADER_BYTES = 4 * HEADER_INTS


def load_model(hf_id):
    """Loads a GPT2LMHeadModel in fp32 eval mode and checks the engine can run it."""
    model = GPT2LMHeadModel.from_pretrained(hf_id).float().eval()
    # transformers 5 maps safetensors in place, and gpt2's wte lands on an odd address
    # (ptr % 64 == 19). Accelerate's GEMV then dies with SIGBUS during generate. Cloning gives
    # every parameter its own aligned buffer; tied weights share one Parameter, so they stay tied.
    for p in model.parameters():
        p.data = p.data.clone()
    check_supported(model)
    return model


def check_supported(model):
    # The header stores only shapes, so every other architecture knob must be the GPT2 default.
    c = model.config
    assert c.activation_function == "gelu_new", c.activation_function
    assert c.layer_norm_epsilon == 1e-5, c.layer_norm_epsilon
    assert c.n_inner in (None, 4 * c.n_embd), c.n_inner
    assert c.n_embd % c.n_head == 0, (c.n_embd, c.n_head)
    assert c.scale_attn_weights and not c.scale_attn_by_inverse_layer_idx
    assert not c.reorder_and_upcast_attn and not c.add_cross_attention
    assert torch.equal(model.lm_head.weight, model.transformer.wte.weight), "LM head must be tied to wte"


def config_of(model):
    c = model.config
    return {"n_layer": c.n_layer, "d_model": c.n_embd, "n_head": c.n_head,
            "vocab": c.vocab_size, "n_ctx": c.n_positions}


def expected_floats(cfg):
    """Total floats of a weights v1 file, the formula in docs/FORMATS.md."""
    v, t, d, n = cfg["vocab"], cfg["n_ctx"], cfg["d_model"], cfg["n_layer"]
    return v * d + t * d + n * (12 * d * d + 13 * d) + 2 * d


def header(cfg):
    h = np.zeros(HEADER_INTS, dtype="<i4")
    h[:8] = [MAGIC, VERSION, DTYPE_FP32, cfg["n_layer"], cfg["d_model"], cfg["n_head"],
             cfg["vocab"], cfg["n_ctx"]]
    return h


def weight_tensors(model):
    """Yields (name, array) in weights v1 order, little endian fp32, linear weights [out, in].

    A generator, so exporting ProtGPT2 never holds a second copy of every tensor.
    """
    cfg = config_of(model)
    v, t, d = cfg["vocab"], cfg["n_ctx"], cfg["d_model"]

    def take(name, param, shape, transpose=False):
        a = param.detach().cpu().numpy()
        # HF Conv1D stores [in, out]; transposing gives our [out, in].
        a = np.ascontiguousarray(a.T if transpose else a, dtype="<f4")
        assert a.shape == shape, (name, a.shape, shape)
        return name, a

    tr = model.transformer
    yield take("wte", tr.wte.weight, (v, d))
    yield take("wpe", tr.wpe.weight, (t, d))
    for i, b in enumerate(tr.h):
        p = f"h{i}."
        yield take(p + "ln1_w", b.ln_1.weight, (d,))
        yield take(p + "ln1_b", b.ln_1.bias, (d,))
        yield take(p + "qkv_w", b.attn.c_attn.weight, (3 * d, d), transpose=True)
        yield take(p + "qkv_b", b.attn.c_attn.bias, (3 * d,))
        yield take(p + "proj_w", b.attn.c_proj.weight, (d, d), transpose=True)
        yield take(p + "proj_b", b.attn.c_proj.bias, (d,))
        yield take(p + "ln2_w", b.ln_2.weight, (d,))
        yield take(p + "ln2_b", b.ln_2.bias, (d,))
        yield take(p + "fc_w", b.mlp.c_fc.weight, (4 * d, d), transpose=True)
        yield take(p + "fc_b", b.mlp.c_fc.bias, (4 * d,))
        yield take(p + "fcproj_w", b.mlp.c_proj.weight, (d, 4 * d), transpose=True)
        yield take(p + "fcproj_b", b.mlp.c_proj.bias, (d,))
    yield take("lnf_w", tr.ln_f.weight, (d,))
    yield take("lnf_b", tr.ln_f.bias, (d,))
