from __future__ import annotations

from pathlib import Path

import regex

# directly taken from the tiktoken library by openai https://github.com/openai/tiktoken
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
_TOKEN_RE = regex.compile(pat_str)


def split_text(text: str, pattern: str | None = None) -> list[str]:
    if pattern is not None:
        return regex.compile(pattern).findall(text)
    return _TOKEN_RE.findall(text)


def split_text_file(path: str | Path, pattern: str | None = None) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if pattern is not None:
        return regex.compile(pattern).findall(text)
    return _TOKEN_RE.findall(text)
