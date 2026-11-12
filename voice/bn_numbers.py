"""Numbers -> Bangla words, so the Bangla voice (which only reads letters) can say them.

    2024 -> দুই হাজার চব্বিশ      3.5 -> তিন দশমিক পাঁচ      1,50,000 -> এক লাখ পঞ্চাশ হাজার
"""
from __future__ import annotations

import re

_0_99 = (
    "শূন্য এক দুই তিন চার পাঁচ ছয় সাত আট নয় "
    "দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ "
    "বিশ একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আটাশ উনত্রিশ "
    "ত্রিশ একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ ছত্রিশ সাঁইত্রিশ আটত্রিশ উনচল্লিশ "
    "চল্লিশ একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ সাতচল্লিশ আটচল্লিশ উনপঞ্চাশ "
    "পঞ্চাশ একান্ন বাহান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন আটান্ন উনষাট "
    "ষাট একষট্টি বাষট্টি তেষট্টি চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি উনসত্তর "
    "সত্তর একাত্তর বাহাত্তর তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর আটাত্তর উনআশি "
    "আশি একাশি বিরাশি তিরাশি চুরাশি পঁচাশি ছিয়াশি সাতাশি আটাশি উননব্বই "
    "নব্বই একানব্বই বিরানব্বই তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই আটানব্বই নিরানব্বই"
).split()
assert len(_0_99) == 100

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def int_to_words(n: int) -> str:
    if n < 0:
        return "মাইনাস " + int_to_words(-n)
    if n < 100:
        return _0_99[n]
    parts = []
    for size, name in ((10_000_000, "কোটি"), (100_000, "লাখ"), (1000, "হাজার"), (100, "শো")):
        if n >= size:
            q, n = divmod(n, size)
            head = int_to_words(q) if size != 100 else _0_99[q]
            parts.append(head + name if size == 100 else f"{head} {name}")
    if n:
        parts.append(_0_99[n])
    return " ".join(parts)


def number_to_words(text: str) -> str:
    """One number as written ('2,500', '3.75', '০৭') -> words."""
    text = text.translate(_BN_DIGITS).replace(",", "")
    if "." in text:
        whole, frac = text.split(".", 1)
        return f"{int_to_words(int(whole or 0))} দশমিক " + " ".join(_0_99[int(d)] for d in frac if d.isdigit())
    if len(text) > 1 and text.startswith("0"):           # phone numbers, codes: digit by digit
        return " ".join(_0_99[int(d)] for d in text)
    return int_to_words(int(text))


_NUMBER = re.compile(r"[0-9০-৯]+(?:[,][0-9০-৯]{2,3})*(?:\.[0-9০-৯]+)?")


def spell_numbers(text: str) -> str:
    """Every number in the text -> Bangla words; % is read as শতাংশ."""
    text = re.sub(r"([0-9০-৯])\s*%", r"\1 শতাংশ", text)
    return _NUMBER.sub(lambda m: " " + number_to_words(m.group(0)) + " ", text)
