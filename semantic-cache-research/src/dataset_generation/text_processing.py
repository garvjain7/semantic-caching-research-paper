"""Shared lexical tokenization for corpus filters and overlap measures."""

import re


TOKENIZER_ID = "unicode_word_tokens_lowercase_v1"
_WORD_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize_question(text: str) -> tuple[str, ...]:
    """Return lowercase Unicode word/number tokens, excluding punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(_WORD_TOKEN_PATTERN.findall(text.lower()))


def token_jaccard(text_a: str, text_b: str) -> float:
    """Token-set Jaccard similarity; empty/empty is defined as zero."""
    tokens_a = set(tokenize_question(text_a))
    tokens_b = set(tokenize_question(text_b))
    union_size = len(tokens_a | tokens_b)
    return len(tokens_a & tokens_b) / union_size if union_size else 0.0


def normalized_edit_distance(text_a: str, text_b: str) -> float:
    """Character-level Levenshtein distance divided by the longer text length."""
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        raise TypeError("edit-distance inputs must be strings")
    left, right = text_a.casefold(), text_b.casefold()
    if not left and not right:
        return 0.0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, start=1):
        current = [row]
        for column, char_right in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1] / max(len(left), len(right))
