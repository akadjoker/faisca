#!/usr/bin/env python3
"""Differential test of the tokenizer against llama.cpp.

    python3 tests/compare_tokenizer.py <tokenizer_dump> <llama_tokenizer_dump> <model.gguf>
                                       [--files a.txt b.cpp ...] [--random N] [--seed S] [--special "<tok>" ...]

Both tools tokenize the same cases with the same GGUF and print the token ids
and two detokenizations. Every line must be identical.

Cases: each file whole, by paragraph and by line; then N random strings mixing
scripts, digits of several kinds, every whitespace, apostrophe forms, unassigned
code points and malformed UTF-8, with the --special strings inserted at random.
"""

import argparse
import random
import struct
import subprocess
import sys
import tempfile

ASCII_WORDS = ["the", "The", "of", "and", "Hello", "world", "tokenizer", "don", "I", "you", "we", "they", "it",
               "x", "foo_bar", "CamelCase", "http", "www", "e", "a"]
APOSTROPHES = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'", "''", "'x", "'r", "'l", "'v", " 's", " 't",
               " ' ", " '", "' ", " 're", " 've", " 'll", " 'm", " 'd"]
PUNCTUATION = list("!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~")
SPACES = [" ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", " \n", "\n ", "\x0b", "\x0c", "\x1c", "\x1f", "\x85",
          "\xa0", " ", " ", " ", " ", " ", " ", " ", "　", "​", "﻿"]
DIGITS = list("0123456789") + ["٣", "۵", "३", "²", "½", "Ⅷ", "１", "〇",
                               "①", "\U0001d7ce", "12", "2024", "3.14", "1,000", "0x1F"]
SCRIPTS = [
    (0x00C0, 0x024F), (0x0370, 0x03FF), (0x0400, 0x04FF), (0x0530, 0x058F), (0x05D0, 0x05EA), (0x0620, 0x064A),
    (0x0900, 0x097F), (0x0E00, 0x0E7F), (0x3040, 0x30FF), (0x4E00, 0x4FFF), (0xAC00, 0xAD00), (0x0300, 0x036F),
    (0x2000, 0x206F), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x1F300, 0x1F64F), (0x1F900, 0x1F9FF), (0x10000, 0x1007F),
    (0xE000, 0xE010), (0x0378, 0x0379), (0xFFF0, 0xFFFF), (0x10FFF0, 0x10FFFF), (0x0080, 0x00BF), (0x0100, 0x0148),
]
BAD_BYTES = [b"\x80", b"\xbf", b"\xc0", b"\xc3", b"\xc3\x28", b"\xe2\x82", b"\xe2\x28\xa1", b"\xf0\x9f\x98",
             b"\xf8\x88\x80\x80\x80", b"\xf4\x90\x80\x80", b"\xf7\xbf\xbf\xbf", b"\xff", b"\xfe", b"\xed\xa0\x80", b"\xc0\xaf", b"\xe0\x80\xaf", b"\xf0\x80\x80\xaf"]


def random_case(rng: random.Random, specials: list[str]) -> bytes:
    parts = []
    for _ in range(rng.randint(1, 24)):
        kind = rng.random()
        if kind < 0.22:
            parts.append(rng.choice(ASCII_WORDS).encode())
        elif kind < 0.40:
            parts.append(rng.choice(SPACES).encode("utf-8"))
        elif kind < 0.50:
            parts.append(rng.choice(APOSTROPHES).encode())
        elif kind < 0.60:
            parts.append(rng.choice(PUNCTUATION).encode() * rng.randint(1, 3))
        elif kind < 0.70:
            parts.append(rng.choice(DIGITS).encode("utf-8"))
        elif kind < 0.90:
            low, high = rng.choice(SCRIPTS)
            text = "".join(chr(rng.randint(low, high)) for _ in range(rng.randint(1, 4)))
            parts.append(text.encode("utf-8", "surrogatepass"))
        elif kind < 0.95:
            parts.append(rng.choice(BAD_BYTES))
        elif specials:
            parts.append(rng.choice(specials).encode("utf-8"))
        else:
            parts.append(bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 3))))
    return b"".join(parts)


def file_cases(path: str) -> list[bytes]:
    data = open(path, "rb").read()
    cases = [data]
    cases += [part for part in data.split(b"\n\n") if part]
    cases += [line for line in data.split(b"\n") if line]
    return cases


def run(tool: str, model: str, cases_path: str) -> list[str]:
    result = subprocess.run([tool, model, cases_path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"{tool}: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout.decode("ascii").splitlines()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ours")
    parser.add_argument("reference")
    parser.add_argument("model")
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--random", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--special", nargs="*", default=[])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = []
    for path in args.files:
        for text in file_cases(path):
            cases.append((rng.randint(0, 3), text))
    for _ in range(args.random):
        cases.append((rng.randint(0, 3), random_case(rng, args.special)))
    cases += [(flags, b"") for flags in range(4)]

    with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
        for flags, text in cases:
            handle.write(struct.pack("<II", flags, len(text)))
            handle.write(text)
        handle.flush()
        ours = run(args.ours, args.model, handle.name)
        reference = run(args.reference, args.model, handle.name)

    if len(ours) != len(cases) or len(reference) != len(cases):
        print(f"FAIL {args.model}: {len(cases)} cases, {len(ours)} lines from faisca, {len(reference)} from llama.cpp")
        return 1

    mismatches = 0
    reference_failed = 0
    tokens = 0
    for (flags, text), mine, theirs in zip(cases, ours, reference):
        if theirs == "FAILED":
            reference_failed += 1
            continue
        tokens += len(theirs.split("|")[0].split()) - 1
        if mine != theirs:
            mismatches += 1
            if mismatches <= 5:
                print(f"  flags {flags} text {text[:120]!r}")
                print(f"    llama.cpp: {theirs[:300]}")
                print(f"    faisca:    {mine[:300]}")

    status = "ok  " if mismatches == 0 else "FAIL"
    print(f"{status} {args.model}: {len(cases)} cases, {tokens} tokens, {mismatches} mismatches, "
          f"{reference_failed} cases llama.cpp could not tokenize")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
