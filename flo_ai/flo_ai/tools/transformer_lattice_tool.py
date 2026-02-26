from __future__ import annotations

import os
import subprocess
import ctypes
from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

from .flo_tool import flotool


class _TransformerArgs(BaseModel):
    embeddings: List[List[float]]
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 16

    # dynamic fields
    seq_len: int | None = None
    d_model: int | None = None

    @model_validator(mode="after")
    def _set_shapes(self):
        if len(self.embeddings) == 0:
            raise ValueError("embeddings cannot be empty")
        row_lens = {len(r) for r in self.embeddings}
        if len(row_lens) != 1:
            raise ValueError("All embedding rows must have the same length")
        d_model = row_lens.pop()
        seq_len = len(self.embeddings)
        if d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.seq_len = seq_len
        self.d_model = d_model
        return self


def _compile_shared_library(lib_path: Path, source_path: Path):
    """Compile the lattice shared library on-the-fly if it is missing."""
    compile_cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        str(source_path),
        "-lm",
        "-o",
        str(lib_path),
    ]
    subprocess.check_call(compile_cmd)


@flotool(
    name="transformer_lattice",
    description="Run the Transformer Lattice (C implementation) on a batch of token embeddings.",
    argument_contract=_TransformerArgs,
)
def transformer_lattice_tool(args: _TransformerArgs):
    """Executes the transformer lattice on the provided embedding matrix.

    The embeddings should be provided as a list of *row-major* lists (shape: seq_len × d_model).
    Returns the transformed embeddings with the same shape.
    """

    # Locate / build shared library
    current_dir = Path(__file__).resolve().parent
    lib_path = current_dir / "liblattice.so"
    source_path = Path(os.environ.get("LATTICE_SRC", "./transformer_lattice.c")).resolve()

    if not lib_path.exists():
        if not source_path.exists():
            raise FileNotFoundError(
                f"Neither shared library {lib_path} nor source file {source_path} could be found."
            )
        _compile_shared_library(lib_path, source_path)

    lib = ctypes.CDLL(str(lib_path))

    # Configure argtypes / restype
    lib.lattice_forward_api.argtypes = [
        ctypes.c_int,  # seq_len
        ctypes.c_int,  # d_model
        ctypes.c_int,  # n_heads
        ctypes.c_int,  # d_ff
        ctypes.c_int,  # n_layers
        ctypes.POINTER(ctypes.c_float),  # data
    ]
    lib.lattice_forward_api.restype = None

    # Prepare data buffer
    arr = np.array(args.embeddings, dtype=np.float32, order="C")
    flat_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Run the lattice
    lib.lattice_forward_api(
        args.seq_len,
        args.d_model,
        args.n_heads,
        args.d_ff,
        args.n_layers,
        flat_ptr,
    )

    return arr.tolist()