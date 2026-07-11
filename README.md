# bpe_rs

`bpe_rs` is a high-performance Byte Pair Encoding (BPE) tokenizer implemented in Rust and exposed to Python via PyO3. It is the tokenizer extracted from the [`ground_llm`](https://github.com/vvaibhavv11/ground_llm) project so it can be reused independently across LLM pipelines.

## Features

- **Fast BPE Training**: Efficiently learns merge rules from training data.
- **Parallelized Statistics**: Uses the `rayon` crate to parallelize pair-frequency computation.
- **Incremental Updates**: Pair counts are updated on each merge instead of a full recount.
- **Python Bindings**: Seamless integration with the rest of an LLM pipeline via `maturin`/PyO3.

## Installation

This project uses `maturin` to build the Rust extension as a Python package.

**Prerequisites:**
- Rust toolchain (`cargo`, `rustc`)
- Python 3.8+
- `maturin` (`pip install maturin`)

**Build and install (local development):**
```bash
maturin develop
```

## Usage

### Training the tokenizer
```python
import bpe_rs

chunks = ["hello world", "hello rust", "bpe tokenizer is fast"]
merges = bpe_rs.encode_train(chunks)
print(f"Learned {len(merges)} merge rules.")
```
Training writes `data/merges_record.json` and `data/vocab_list.json` into a `data/` directory next to the CWD.

### Encoding text
```python
import bpe_rs
import json

with open("data/8000/merges_record.json", "r") as f:
    merges_data = json.load(f)

merges = [((tuple(m["pair"])), m["id"]) for m in merges_data]
tokens = bpe_rs.encode(["hello rust"], merges)
print(f"Tokens: {tokens}")
```

### Decoding tokens
```python
import bpe_rs
import json

with open("data/8000/vocab_list.json", "r") as f:
    vocab_list_data = json.load(f)

vocab_list = {item["id"]: bytes(item["bytes"]) for item in vocab_list_data}
decoded_text = bpe_rs.decode_string([101, 256, 102], vocab_list)
print(f"Decoded text: {decoded_text}")
```

## Precomputed vocab runs

The `data/` directory ships with two precomputed vocab/merge runs:

| Run            | Path                   | Vocab size | Notes                  |
| -------------- | ---------------------- | ---------- | ---------------------- |
| 8,000 tokens   | `data/8000/*.json`     | 8,000      | Canonical default run   |
| Full run       | `data/full/*.json`     | varies     | Larger experimental run |

## API reference

```python
def build_info() -> str: ...
def encode_train(s: list[str]) -> list[tuple[tuple[int, int], int]]: ...
def encode(s: list[str], merges: list[tuple[tuple[int, int], int]]) -> list[int]: ...
def decode_string(ids: list[int], vocab_list: dict[int, list[int]]) -> str: ...
```

## Project structure

- `src/lib.rs`: Python module definition and function exports.
- `src/encoder.rs`: Core BPE logic.
- `Cargo.toml`: Rust dependencies.
- `pyproject.toml`: Python project configuration.
- `bpe_rs.pyi`: Type hints for the Python module.
- `data/`: Precomputed merges + vocab JSON files.
- `tests/test_bindings.py`: Python driver test.
