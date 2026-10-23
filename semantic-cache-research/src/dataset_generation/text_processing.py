"""Shared lexical tokenization for corpus filters and overlap measures."""

import re


TOKENIZER_ID = "unicode_word_tokens_lowercase_v1"
_WORD_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize_question(text: str) -> tuple[str, ...]:
    """Return lowercase Unicode word/number tokens, excluding punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(_WORD_TOKEN_PATTERN.findall(text.lower()))
