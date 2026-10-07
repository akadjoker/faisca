#!/usr/bin/env python3
"""Writes a vocabulary-only GGUF from GPT-2 style files, to test the tokenizer.

    python3 tests/make_tokenizer_gguf.py <encoder.json> <vocab.bpe> <out.gguf> --pre gpt-2|smollm [--special] [--bos]

--special adds control and user defined tokens, an empty text and a token
outside the byte alphabet. --bos sets add_bos_token and
add_eos_token.
"""

import argparse
import json
import sys

from gguf import GGUFWriter

NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder")
    parser.add_argument("merges")
    parser.add_argument("out")
    parser.add_argument("--pre", required=True)
    parser.add_argument("--special", action="store_true")
    parser.add_argument("--bos", action="store_true")
    args = parser.parse_args()

    encoder = json.load(open(args.encoder, encoding="utf-8"))
    tokens = [text for text, _ in sorted(encoder.items(), key=lambda item: item[1])]
    types = [NORMAL] * len(tokens)
    types[-1] = CONTROL
    merges = [line for line in open(args.merges, encoding="utf-8").read().split("\n")[1:] if line]

    bos = len(tokens) - 1
    eos = len(tokens) - 1
    if args.special:
        extra = [
            ("<|im_start|>", CONTROL),
            ("<|im_end|>", CONTROL),
            ("<|im_start|>assistant", USER_DEFINED),
            ("<tool>", USER_DEFINED),
            ("</tool>", USER_DEFINED),
            ("<unk>", UNKNOWN),
            ("<unused0>", UNUSED),
            ("<0x41>", BYTE),
            ("", NORMAL),
            ("日本", NORMAL),
            ("aa", USER_DEFINED),
            ("ab", USER_DEFINED),
            ("ba", USER_DEFINED),
            ("ĠĠ", CONTROL),
        ]
        bos = len(tokens)
        eos = len(tokens) + 1
        for text, kind in extra:
            tokens.append(text)
            types.append(kind)

    writer = GGUFWriter(args.out, "llama")
    writer.add_name("tokenizer test vocabulary")
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(args.pre)
    writer.add_token_list([token.encode("utf-8") for token in tokens])
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(bos)
    writer.add_eos_token_id(eos)
    if args.bos:
        writer.add_add_bos_token(True)
        writer.add_add_eos_token(True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
