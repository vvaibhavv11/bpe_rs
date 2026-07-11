from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import regex as re

import bpe_rs

ROOT = Path(__file__).resolve().parent.parent
Merge = Tuple[Tuple[int, int], int]
VocabList = Dict[int, List[int]]
EXPECTED_EXPORTS = (
    "build_info",
    "encode_train",
    "encode",
    "decode_string",
)
# directly taken from the tiktoken library by openai https://github.com/openai/tiktoken but tiktoken does not contain the training code
pat_str = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
TOKEN_RE = re.compile(pat_str)


def summarize_text(text: str, preview: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= preview:
        return compact
    return f"{compact[:preview]}..."


def split_text_file(text_file: str | Path) -> list[str]:
    return TOKEN_RE.findall(Path(text_file).read_text(encoding="utf-8"))


def load_merges_record(path: str | Path) -> list[Merge]:
    raw_merges: list[dict[str, Any]] = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    return [
        ((int(entry["pair"][0]), int(entry["pair"][1])), int(entry["id"]))
        for entry in raw_merges
    ]


def load_vocab_list(path: str | Path) -> VocabList:
    raw_vocab: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(entry["id"]): [int(byte) for byte in entry["bytes"]] for entry in raw_vocab
    }


def check_exports() -> int:
    missing = [name for name in EXPECTED_EXPORTS if not hasattr(bpe_rs, name)]
    if missing:
        print(f"missing exports: {missing}")
        return 1
    print("exports OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test bpe_rs bindings")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset" / "tokebnizer_dataset.txt",
        help="text file to train on",
    )
    args = parser.parse_args()

    rc = check_exports()
    if rc:
        return rc

    print(f"build_info={bpe_rs.build_info()}")

    if args.dataset.exists():
        split_text = split_text_file(args.dataset)
        merges = bpe_rs.encode_train(split_text)

        merges_file = ROOT / "data" / "merges_record.json"
        vocab_file = ROOT / "data" / "vocab_list.json"
        saved_merges = json.loads(merges_file.read_text(encoding="utf-8"))
        saved_vocab = json.loads(vocab_file.read_text(encoding="utf-8"))

        print(f"merge_count={len(saved_merges)}")
        print(f"vocab_count={len(saved_vocab)}")
        print(f"vocab_size=8000")
        if saved_merges:
            print(f"first_merge={saved_merges[0]}")
            print(f"last_merge={saved_merges[-1]}")
        if saved_vocab:
            print(f"first_vocab={saved_vocab[0]}")
            print(f"last_vocab={saved_vocab[-1]}")
    else:
        print(f"dataset not found at {args.dataset}; skipping train step")

    return 0


if __name__ == "__main__":
    sys.exit(main())
