#!/bin/sh
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "== release build"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$ROOT/build" -j > /dev/null
ctest --test-dir "$ROOT/build" --output-on-failure

echo "== sanitizer build (ASan + UBSan)"
cmake -S "$ROOT" -B "$ROOT/build-sanitize" -DCMAKE_BUILD_TYPE=Debug -DFAISCA_SANITIZE=ON > /dev/null
cmake --build "$ROOT/build-sanitize" -j > /dev/null
ctest --test-dir "$ROOT/build-sanitize" --output-on-failure

if [ "$#" -gt 0 ]
then
    echo "== real models against gguf-py"
    python3 "$ROOT/tests/compare_with_gguf_py.py" "$ROOT/build/gguf_dump" "$@"
fi

echo "all tests passed"
