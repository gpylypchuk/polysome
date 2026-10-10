"""Tensor bundle v1 (docs/FORMATS.md): named arrays for golden data and op fixtures."""
import os
import struct

import numpy as np

MAGIC = 20261010
VERSION = 1
DTYPES = {0: np.dtype("<f4"), 1: np.dtype("<i4"), 2: np.dtype("<f8")}
CODES = {dt.str: code for code, dt in DTYPES.items()}


def write(path, tensors):
    """Writes {name: array} in insertion order. Arrays must be float32, int32 or float64."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<3i", MAGIC, VERSION, len(tensors)))
        for name, a in tensors.items():
            a = np.ascontiguousarray(a)
            if a.dtype.str not in CODES:
                raise ValueError(f"{name}: unsupported dtype {a.dtype}, convert to f32, i32 or f64")
            raw = name.encode()
            f.write(struct.pack("<i", len(raw)) + raw)
            f.write(struct.pack("<2i", CODES[a.dtype.str], a.ndim))
            f.write(struct.pack(f"<{a.ndim}q", *a.shape))
            f.write(a.tobytes())
    os.replace(tmp, path)


def read(path):
    """Reads a bundle back into {name: array}."""
    with open(path, "rb") as f:
        buf = f.read()
    magic, version, count = struct.unpack_from("<3i", buf, 0)
    assert (magic, version) == (MAGIC, VERSION), (magic, version)
    off, out = 12, {}
    for _ in range(count):
        (name_len,) = struct.unpack_from("<i", buf, off)
        name = buf[off + 4:off + 4 + name_len].decode()
        off += 4 + name_len
        code, ndim = struct.unpack_from("<2i", buf, off)
        shape = struct.unpack_from(f"<{ndim}q", buf, off + 8)
        off += 8 + 8 * ndim
        dt = DTYPES[code]
        n = int(np.prod(shape))
        out[name] = np.frombuffer(buf, dtype=dt, count=n, offset=off).reshape(shape)
        off += n * dt.itemsize
    assert off == len(buf), f"{len(buf) - off} trailing bytes"
    return out
