# bpe_rs

[![PyPI](https://img.shields.io/pypi/v/bpe_rs.svg)](https://pypi.org/project/bpe_rs/)
[![Python Versions](https://img.shields.io/pypi/pyversions/bpe_rs.svg)](https://pypi.org/project/bpe_rs/)
[![License](https://img.shields.io/github/license/vvaibhavv11/bpe_rs)](LICENSE)
[![CI](https://github.com/vvaibhavv11/bpe_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/vvaibhavv11/bpe_rs/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange)](https://www.rust-lang.org)

A high-performance Byte Pair Encoding (BPE) tokenizer implemented in Rust with Python bindings via PyO3. Extracted from [`ground_llm`](https://github.com/vvaibhavv11/ground_llm) for standalone reuse across LLM pipelines.

## Features

- **Fast BPE Training** — Efficiently learns merge rules from training data.
- **Parallelized Statistics** — Uses `rayon` to parallelize pair-frequency computation.
- **Incremental Updates** — Pair counts are updated on each merge instead of a full recount.
- **Python Bindings** — Seamless integration with LLM pipelines via PyO3/maturin.
- **Tiktoken-Compatible Pre-Tokenizer** — GPT-family regex splitting.

## Installation

```bash
pip install bpe_rs
```

**Requirements:** Python 3.8+

## Quick Start

```python
from bpe_rs import BPETokenizer

tok = BPETokenizer()
tok.train("hello world hello rust bpe tokenizer is fast")

tokens = tok.encode("hello rust")
print(tokens)

text = tok.decode(tokens)
print(text)
```

## API

### `BPETokenizer`

```python
class BPETokenizer:
    def __init__(self, pattern: str | None = None): ...
    def train(self, text: str | list[str]) -> None: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str, pattern: str | None = None) -> BPETokenizer: ...
```

### Low-level functions

```python
def build_info() -> str: ...
def encode_train(s: list[str]) -> list[tuple[tuple[int, int], int]]: ...
def encode(s: list[str], merges: list[tuple[tuple[int, int], int]]) -> list[int]: ...
def decode_string(ids: list[int], vocab_list: dict[int, list[int]]) -> str: ...
```

### Pre-tokenization

```python
from bpe_rs import pre_tokenize

chunks = pre_tokenize.split_text("hello world! this is a test.")
chunks = pre_tokenize.split_text_file("dataset.txt")
```

## Examples

### Training

```python
from bpe_rs import BPETokenizer

tok = BPETokenizer()
tok.train(["hello world", "hello rust", "bpe tokenizer is fast"])
print(f"Vocabulary size: {len(tok._merges) + 256}")
```

### Encoding and decoding

```python
from bpe_rs import BPETokenizer

tok = BPETokenizer()
tok.train("the cat sat on the mat")

ids = tok.encode("the cat sat")
print(f"Token IDs: {ids}")

reconstructed = tok.decode(ids)
print(f"Reconstructed: {reconstructed}")
```

### Saving and loading

```python
from bpe_rs import BPETokenizer

# Train and save
tok = BPETokenizer()
tok.train("large training corpus goes here")
tok.save("my_tokenizer/")

# Load and reuse
tok2 = BPETokenizer.load("my_tokenizer/")
result = tok2.decode(tok2.encode("hello world"))
print(result)
```

### Using a custom regex pattern

```python
from bpe_rs import BPETokenizer

tok = BPETokenizer(pattern=r"\w+")
tok.train("hello world")
```

## Precomputed vocab runs

| Run          | Path                  | Vocab size | Notes                  |
| ------------ | --------------------- | ---------- | ---------------------- |
| 8,000 tokens | `data/8000/*.json`    | 8,000      | Default run            |
| Full run     | `data/full/*.json`    | ~50,000    | Larger experimental    |

## Development

```bash
git clone https://github.com/vvaibhavv11/bpe_rs.git
cd bpe_rs

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dev dependencies
uv sync

# Build and install in development mode
maturin develop

# Run tests
cargo test                 # Rust unit tests
python tests/test_bindings.py  # Python integration tests
```

## Release

1. Update `version` in `Cargo.toml`
2. Commit: `git commit -m "bump version to x.y.z"`
3. Tag: `git tag vx.y.z`
4. Push: `git push && git push --tags`
5. Create a GitHub Release from the tag
6. The `publish.yml` workflow builds multi-platform wheels and publishes to PyPI automatically.

## License

MIT — see [LICENSE](LICENSE).

## Project structure

```
bpe_rs/
├── src/
│   ├── lib.rs          # PyO3 module definition
│   └── tokenizer.rs    # Core BPE logic
├── python/bpe_rs/
│   ├── __init__.py     # Public API
│   ├── __init__.pyi    # Type stubs
│   ├── py.typed        # PEP 561 marker
│   ├── tokenizer.py    # Python BPETokenizer class
│   └── pre_tokenize.py # Regex pre-tokenizer
├── data/               # Precomputed vocab files
├── tests/
├── Cargo.toml
├── pyproject.toml
└── README.md
```