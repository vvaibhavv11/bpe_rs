from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from bpe_rs._bpe_rs import build_info, decode_string, encode, encode_train
from bpe_rs.pre_tokenize import split_text

Merge = Tuple[Tuple[int, int], int]
VocabList = Dict[int, List[int]]


def _build_vocab_list(merges: List[Merge]) -> VocabList:
    vocab: VocabList = {}
    for idx in range(256):
        vocab[idx] = [idx]
    for (p0, p1), id_ in merges:
        vocab[id_] = vocab[p0] + vocab[p1]
    return vocab


class BPETokenizer:
    def __init__(self, pattern: str | None = None):
        self._pattern = pattern
        self._merges: List[Merge] | None = None
        self._vocab: VocabList | None = None

    def train(self, text: str | List[str]) -> None:
        if isinstance(text, str):
            text = [text]

        chunks: List[str] = []
        for t in text:
            chunks.extend(split_text(t, pattern=self._pattern))

        if not chunks:
            raise ValueError("no chunks produced from the given text")

        self._merges = encode_train(chunks)
        self._vocab = _build_vocab_list(self._merges)

    def encode(self, text: str) -> List[int]:
        if self._merges is None:
            raise RuntimeError("tokenizer has not been trained yet")

        chunks = split_text(text, pattern=self._pattern)
        return encode(chunks, self._merges)

    def decode(self, ids: List[int]) -> str:
        if self._vocab is None:
            raise RuntimeError("tokenizer has not been trained yet")

        return decode_string(ids, self._vocab)

    def save(self, path: str) -> None:
        if self._merges is None:
            raise RuntimeError("tokenizer has not been trained yet")

        os.makedirs(path, exist_ok=True)
        merges_path = Path(path) / "merges.json"
        merges_data = [
            {"pair": [p0, p1], "id": id_}
            for (p0, p1), id_ in self._merges
        ]
        merges_path.write_text(
            json.dumps(merges_data, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str, pattern: str | None = None) -> BPETokenizer:
        merges_path = Path(path) / "merges.json"
        if not merges_path.exists():
            raise FileNotFoundError(f"{merges_path} not found")

        merges_data = json.loads(merges_path.read_text(encoding="utf-8"))
        merges: List[Merge] = [
            ((entry["pair"][0], entry["pair"][1]), entry["id"])
            for entry in merges_data
        ]

        tok = cls(pattern=pattern)
        tok._merges = merges
        tok._vocab = _build_vocab_list(merges)
        return tok
