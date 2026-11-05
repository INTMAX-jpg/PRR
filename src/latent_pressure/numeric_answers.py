from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Optional


_NUMBER_BODY = (
    r"[-+]?(?:"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"|\.\d+"
    r")(?:[eE][-+]?\d+)?"
)
_FRACTION_OR_NUMBER = rf"{_NUMBER_BODY}(?:\s*/\s*{_NUMBER_BODY})?%?"
_NUMBER_RE = re.compile(rf"(?<![\w.])({_FRACTION_OR_NUMBER})(?![\w.])")
_FINAL_MARKER_RE = re.compile(
    rf"(?is)(?:FINAL[\s_-]*ANSWER|FINAL)\s*(?:IS\s*)?(?::|=|-)?\s*"
    rf"(?:\\boxed\s*\{{\s*)?(\$?\s*{_FRACTION_OR_NUMBER})"
)
_HASH_MARKER_RE = re.compile(rf"(?is)####\s*(\$?\s*{_FRACTION_OR_NUMBER})")
_BOXED_RE = re.compile(r"(?is)\\boxed\s*\{([^{}]+)\}")


@dataclass(frozen=True)
class ExtractedNumber:
    value: Optional[Decimal]
    raw: Optional[str]
    normalized: Optional[str]
    source: str
    marker_found: bool


def canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    if value == value.to_integral_value():
        return format(value.quantize(Decimal(1)), "f")
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_number(text: str) -> Decimal:
    token = str(text).strip()
    token = (
        token.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .strip()
    )
    # GSM8K expects the displayed numeric magnitude for percentage answers
    # (e.g. gold "60"), so "60%" is normalized to 60 rather than 0.60.
    if token.endswith("%"):
        token = token[:-1].strip()
    try:
        if "/" in token:
            left, right = (part.strip() for part in token.split("/", 1))
            with localcontext() as ctx:
                ctx.prec = 50
                denominator = Decimal(right)
                if denominator == 0:
                    raise ValueError("division by zero")
                return Decimal(left) / denominator
        return Decimal(token)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid numeric answer: {text!r}") from exc


def _from_raw(raw: str, source: str, marker_found: bool) -> ExtractedNumber:
    try:
        value = parse_number(raw)
    except ValueError:
        return ExtractedNumber(None, raw, None, source, marker_found)
    return ExtractedNumber(value, raw, canonical_decimal(value), source, marker_found)


def extract_final_number(text: str) -> ExtractedNumber:
    """Extract a model's final numeric answer using deterministic precedence.

    Precedence: explicit FINAL_ANSWER marker, GSM8K #### marker, last boxed
    expression, then the final numeric token in the response.
    """
    normalized_text = (
        str(text)
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    matches = list(_FINAL_MARKER_RE.finditer(normalized_text))
    if matches:
        return _from_raw(matches[-1].group(1), "final_marker", True)

    matches = list(_HASH_MARKER_RE.finditer(normalized_text))
    if matches:
        return _from_raw(matches[-1].group(1), "hash_marker", True)

    boxed = list(_BOXED_RE.finditer(normalized_text))
    if boxed:
        inner = boxed[-1].group(1)
        nums = list(_NUMBER_RE.finditer(inner))
        if nums:
            return _from_raw(nums[-1].group(1), "boxed", True)

    nums = list(_NUMBER_RE.finditer(normalized_text))
    if nums:
        return _from_raw(nums[-1].group(1), "last_number", False)

    return ExtractedNumber(None, None, None, "unparsed", False)


def numerically_equal(
    left: Optional[Decimal],
    right: Optional[Decimal],
    *,
    rel_tol: Decimal = Decimal("1e-9"),
    abs_tol: Decimal = Decimal("1e-9"),
) -> bool:
    if left is None or right is None:
        return False
    difference = abs(left - right)
    scale = max(Decimal(1), abs(left), abs(right))
    return difference <= max(abs_tol, rel_tol * scale)


def self_test() -> None:
    cases = [
        ("Reasoning.\nFINAL_ANSWER: 1,234", Decimal("1234"), "final_marker"),
        ("Thus \\boxed{-3/4}.", Decimal("-0.75"), "boxed"),
        ("work\n#### 6.25", Decimal("6.25"), "hash_marker"),
        ("The result is 60%.", Decimal("60"), "last_number"),
        ("No numeric result.", None, "unparsed"),
        ("Earlier 12.\nFINAL ANSWER = $2.5e2", Decimal("250"), "final_marker"),
    ]
    for text, expected, source in cases:
        found = extract_final_number(text)
        assert found.value == expected, (text, found, expected)
        assert found.source == source, (text, found.source, source)

